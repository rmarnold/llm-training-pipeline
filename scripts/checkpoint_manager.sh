#!/bin/bash
# scripts/checkpoint_manager.sh
# Checkpoint management utilities: archive, cleanup, list, restore

set -e

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
ARCHIVE_DIR="${ARCHIVE_DIR:-checkpoints_archive}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Checkpoint Manager - Manage training checkpoints"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  list [stage]          List all checkpoints (optionally for a specific stage)"
    echo "  archive <path>        Archive a checkpoint to tar.gz"
    echo "  restore <archive>     Restore a checkpoint from archive"
    echo "  cleanup <stage> [n]   Keep only the latest N checkpoints (default: 3)"
    echo "  disk-usage            Show disk usage by stage"
    echo "  export <path> <dest>  Export checkpoint to a new location"
    echo ""
    echo "Examples:"
    echo "  $0 list pretrain"
    echo "  $0 archive checkpoints/pretrain_final"
    echo "  $0 cleanup pretrain 3"
    echo "  $0 disk-usage"
}

list_checkpoints() {
    local stage=$1

    echo "Checkpoints in ${CHECKPOINT_DIR}:"
    echo ""

    if [ -n "$stage" ]; then
        local dirs=("${CHECKPOINT_DIR}/${stage}" "${CHECKPOINT_DIR}/${stage}_final")
    else
        local dirs=("${CHECKPOINT_DIR}"/*)
    fi

    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)

            # Check for checkpoint subdirectories
            local ckpt_count=$(find "$dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l | tr -d ' ')

            if [ "$ckpt_count" -gt 0 ]; then
                echo -e "${GREEN}$name${NC} ($size)"
                echo "  Intermediate checkpoints: $ckpt_count"

                # List latest 3 checkpoints
                find "$dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | \
                    sed 's/.*checkpoint-//' | sort -n | tail -3 | \
                    while read step; do
                        local ckpt_size=$(du -sh "${dir}/checkpoint-${step}" 2>/dev/null | cut -f1)
                        echo "    - checkpoint-${step} ($ckpt_size)"
                    done
            elif [ -f "$dir/config.json" ]; then
                echo -e "${GREEN}$name${NC} ($size) - Final model"
            fi
        fi
    done

    # List archives if any
    if [ -d "$ARCHIVE_DIR" ]; then
        local archive_count=$(find "$ARCHIVE_DIR" -name "*.tar.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$archive_count" -gt 0 ]; then
            echo ""
            echo "Archived checkpoints in ${ARCHIVE_DIR}: $archive_count"
        fi
    fi
}

archive_checkpoint() {
    local checkpoint_path=$1

    if [ -z "$checkpoint_path" ]; then
        echo -e "${RED}Error: No checkpoint path provided${NC}"
        echo "Usage: $0 archive <checkpoint_path>"
        exit 1
    fi

    if [ ! -d "$checkpoint_path" ]; then
        echo -e "${RED}Error: Checkpoint not found: $checkpoint_path${NC}"
        exit 1
    fi

    mkdir -p "$ARCHIVE_DIR"

    local basename=$(basename "$checkpoint_path")
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_name="${basename}_${timestamp}.tar.gz"
    local archive_path="${ARCHIVE_DIR}/${archive_name}"

    echo "Archiving: $checkpoint_path"
    echo "Destination: $archive_path"

    # Get size before archiving
    local original_size=$(du -sh "$checkpoint_path" | cut -f1)
    echo "Original size: $original_size"

    # Create archive
    tar -czf "$archive_path" -C "$(dirname "$checkpoint_path")" "$basename"

    if [ $? -eq 0 ]; then
        local archive_size=$(du -sh "$archive_path" | cut -f1)
        echo -e "${GREEN}✓ Archive created successfully${NC}"
        echo "  Archive size: $archive_size"
        echo "  Path: $archive_path"
    else
        echo -e "${RED}❌ Archive failed${NC}"
        exit 1
    fi
}

restore_checkpoint() {
    local archive_path=$1
    local restore_dir=${2:-$CHECKPOINT_DIR}

    if [ -z "$archive_path" ]; then
        echo -e "${RED}Error: No archive path provided${NC}"
        echo "Usage: $0 restore <archive.tar.gz> [restore_dir]"
        exit 1
    fi

    if [ ! -f "$archive_path" ]; then
        # Try looking in archive directory
        if [ -f "${ARCHIVE_DIR}/${archive_path}" ]; then
            archive_path="${ARCHIVE_DIR}/${archive_path}"
        else
            echo -e "${RED}Error: Archive not found: $archive_path${NC}"
            exit 1
        fi
    fi

    echo "Restoring: $archive_path"
    echo "Destination: $restore_dir"

    mkdir -p "$restore_dir"
    tar -xzf "$archive_path" -C "$restore_dir"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Checkpoint restored successfully${NC}"
    else
        echo -e "${RED}❌ Restore failed${NC}"
        exit 1
    fi
}

cleanup_checkpoints() {
    local stage=$1
    local keep_n=${2:-3}
    local checkpoint_dir="${CHECKPOINT_DIR}/${stage}"

    if [ -z "$stage" ]; then
        echo -e "${RED}Error: No stage provided${NC}"
        echo "Usage: $0 cleanup <stage> [keep_n]"
        exit 1
    fi

    if [ ! -d "$checkpoint_dir" ]; then
        echo -e "${YELLOW}Warning: No checkpoints found for stage: $stage${NC}"
        exit 0
    fi

    # Find all checkpoint directories
    local checkpoints=($(find "$checkpoint_dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | \
        sed 's/.*checkpoint-//' | sort -n))

    local total=${#checkpoints[@]}

    if [ "$total" -le "$keep_n" ]; then
        echo "Found $total checkpoints, keeping all (threshold: $keep_n)"
        exit 0
    fi

    local to_delete=$((total - keep_n))
    echo "Found $total checkpoints, will delete $to_delete oldest"
    echo ""

    # Delete oldest checkpoints
    local deleted=0
    for step in "${checkpoints[@]}"; do
        if [ "$deleted" -ge "$to_delete" ]; then
            break
        fi

        local ckpt_path="${checkpoint_dir}/checkpoint-${step}"
        local ckpt_size=$(du -sh "$ckpt_path" 2>/dev/null | cut -f1)

        echo -e "${YELLOW}Deleting: checkpoint-${step} ($ckpt_size)${NC}"
        rm -rf "$ckpt_path"
        deleted=$((deleted + 1))
    done

    echo ""
    echo -e "${GREEN}✓ Cleanup complete. Deleted $deleted checkpoints.${NC}"

    # Show remaining
    echo ""
    echo "Remaining checkpoints:"
    for step in "${checkpoints[@]:$to_delete}"; do
        echo "  - checkpoint-${step}"
    done
}

show_disk_usage() {
    echo "Disk Usage by Stage"
    echo "==================="
    echo ""

    local total=0
    for stage in pretrain sft dpo lora; do
        local stage_total=0
        local dirs=("${CHECKPOINT_DIR}/${stage}" "${CHECKPOINT_DIR}/${stage}_final")

        for dir in "${dirs[@]}"; do
            if [ -d "$dir" ]; then
                local size_bytes=$(du -sb "$dir" 2>/dev/null | cut -f1)
                stage_total=$((stage_total + size_bytes))
            fi
        done

        if [ "$stage_total" -gt 0 ]; then
            local size_human=$(numfmt --to=iec-i --suffix=B $stage_total 2>/dev/null || echo "${stage_total} bytes")
            printf "%-15s %s\n" "$stage:" "$size_human"
            total=$((total + stage_total))
        fi
    done

    echo ""
    local total_human=$(numfmt --to=iec-i --suffix=B $total 2>/dev/null || echo "${total} bytes")
    echo "Total: $total_human"

    # Archives
    if [ -d "$ARCHIVE_DIR" ]; then
        local archive_size=$(du -sh "$ARCHIVE_DIR" 2>/dev/null | cut -f1)
        echo "Archives: $archive_size"
    fi
}

export_checkpoint() {
    local source_path=$1
    local dest_path=$2

    if [ -z "$source_path" ] || [ -z "$dest_path" ]; then
        echo -e "${RED}Error: Source and destination required${NC}"
        echo "Usage: $0 export <source_checkpoint> <destination>"
        exit 1
    fi

    if [ ! -d "$source_path" ]; then
        echo -e "${RED}Error: Source checkpoint not found: $source_path${NC}"
        exit 1
    fi

    echo "Exporting: $source_path"
    echo "Destination: $dest_path"

    mkdir -p "$dest_path"
    cp -r "$source_path"/* "$dest_path"/

    if [ $? -eq 0 ]; then
        local size=$(du -sh "$dest_path" | cut -f1)
        echo -e "${GREEN}✓ Export complete ($size)${NC}"
    else
        echo -e "${RED}❌ Export failed${NC}"
        exit 1
    fi
}

# Main command router
case "${1:-}" in
    list)
        list_checkpoints "$2"
        ;;
    archive)
        archive_checkpoint "$2"
        ;;
    restore)
        restore_checkpoint "$2" "$3"
        ;;
    cleanup)
        cleanup_checkpoints "$2" "$3"
        ;;
    disk-usage)
        show_disk_usage
        ;;
    export)
        export_checkpoint "$2" "$3"
        ;;
    -h|--help|help|"")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
