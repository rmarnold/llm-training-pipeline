"""Unified Google Drive access — mounted filesystem or service account API.

Two backends:
  - "mounted": Drive is FUSE-mounted (standard Colab ``drive.mount``).
    Uses ``shutil`` for file operations.
  - "service_account": Uses the Drive v3 REST API with a service-account
    JSON key.  Works headlessly (PyCharm Colab plugin, SSH, CI).
  - "local": No Drive access; backup/restore are no-ops (with a warning
    on first call).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DriveHelper:
    """Unified Drive access — mounted filesystem or service account API."""

    def __init__(
        self,
        mode: str,
        drive_base: str = "",
        credentials_path: Optional[str] = None,
        folder_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        mode : "mounted" | "service_account" | "local"
        drive_base : Filesystem path for mounted mode (e.g. "/content/drive/MyDrive/…").
        credentials_path : Path to service-account JSON key (service_account mode).
        folder_id : Root Drive folder ID (service_account mode).
        """
        if mode not in ("mounted", "service_account", "local"):
            raise ValueError(f"Unknown drive mode: {mode!r}")
        self.mode = mode
        self.drive_base = drive_base
        self._warned_local = False
        self._warned_quota = False

        if mode == "service_account":
            if not credentials_path:
                raise ValueError("credentials_path required for service_account mode")
            if not folder_id:
                raise ValueError("folder_id required for service_account mode")
            # Guard against accidentally passing the JSON key as folder_id
            if folder_id.strip().startswith("{"):
                raise ValueError(
                    "folder_id looks like JSON (service account key?), not a Drive folder ID.\n"
                    "Expected a short ID like '18UpFpUhiNrs2Etha0uFjSGWmj1Ee1SnX'.\n"
                    "Check your DRIVE_FOLDER_ID Colab Secret — it should contain "
                    "only the folder ID from the Google Drive URL."
                )
            self._folder_id = folder_id
            self._service = _build_drive_service(credentials_path)
            # Cache: relative_path -> Drive folder ID
            self._folder_cache: dict[str, str] = {"": folder_id}

            # Validate folder access and detect shared drive
            self._drive_id: Optional[str] = None
            try:
                meta = self._service.files().get(
                    fileId=folder_id,
                    fields="id,name,driveId",
                    supportsAllDrives=True,
                ).execute()
                self._drive_id = meta.get("driveId")
                drive_type = "shared drive" if self._drive_id else "My Drive"
                logger.info(
                    "DriveHelper: folder %r (%s) on %s validated OK",
                    meta.get("name"), folder_id, drive_type,
                )
                if not self._drive_id:
                    logger.warning(
                        "DriveHelper: folder is on personal My Drive, not a "
                        "Shared Drive. Service accounts cannot CREATE new files "
                        "here (no storage quota). Uploads of new files will be "
                        "skipped. To fix: create a Shared Drive (requires Google "
                        "Workspace) and move your folder there."
                    )
            except Exception as e:
                # Truncate folder_id in error to avoid leaking secrets
                safe_id = folder_id[:20] + "..." if len(folder_id) > 20 else folder_id
                raise ValueError(
                    f"Cannot access Drive folder {safe_id!r}: {e}\n"
                    f"Ensure the folder is shared with the service account email."
                ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backup(self, local_path: str, drive_relative_path: str) -> None:
        """Copy *local_path* → Drive at *drive_relative_path*.  Recursive for dirs."""
        if self.mode == "local":
            self._warn_local("backup")
            return

        if not os.path.exists(local_path):
            logger.warning("backup: local path %s does not exist, skipping", local_path)
            return

        if self.mode == "mounted":
            dst = os.path.join(self.drive_base, drive_relative_path)
            _copy_local(local_path, dst)
        else:
            try:
                self._api_upload(local_path, drive_relative_path)
            except Exception as e:
                if "storageQuotaExceeded" in str(e):
                    if not self._warned_quota:
                        logger.warning(
                            "Drive backup skipped — service account has no storage "
                            "quota on personal My Drive. To enable backups, move "
                            "your folder to a Shared Drive (requires Google Workspace). "
                            "Training data is safe on the local VM."
                        )
                        self._warned_quota = True
                else:
                    raise

    def restore(self, drive_relative_path: str, local_path: str) -> None:
        """Copy Drive → *local_path*.  Recursive for dirs."""
        if self.mode == "local":
            self._warn_local("restore")
            return

        if self.mode == "mounted":
            src = os.path.join(self.drive_base, drive_relative_path)
            if os.path.exists(src):
                _copy_local(src, local_path)
            else:
                logger.info("restore: %s not found on Drive, skipping", src)
        else:
            self._api_download(drive_relative_path, local_path)

    def ensure_dir(self, drive_relative_path: str) -> None:
        """Create directory on Drive if it doesn't exist."""
        if self.mode == "local":
            return
        if self.mode == "mounted":
            os.makedirs(os.path.join(self.drive_base, drive_relative_path), exist_ok=True)
        else:
            try:
                self._resolve_folder(drive_relative_path, create=True)
            except Exception as e:
                if "storageQuotaExceeded" in str(e):
                    if not self._warned_quota:
                        logger.warning(
                            "Drive ensure_dir skipped — service account has no "
                            "storage quota on personal My Drive."
                        )
                        self._warned_quota = True
                else:
                    raise

    # ------------------------------------------------------------------
    # Internals — mounted helpers
    # ------------------------------------------------------------------

    def _warn_local(self, op: str) -> None:
        if not self._warned_local:
            logger.warning(
                "DriveHelper in 'local' mode — %s is a no-op. "
                "Data is NOT backed up to Google Drive.",
                op,
            )
            self._warned_local = True

    # ------------------------------------------------------------------
    # Internals — service-account API
    # ------------------------------------------------------------------

    def _list_files(self, q: str, fields: str = "files(id)") -> list[dict]:
        """Wrapper around files().list() with correct shared-drive params."""
        kwargs: dict = {
            "q": q,
            "fields": fields,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }
        if self._drive_id:
            kwargs["corpora"] = "drive"
            kwargs["driveId"] = self._drive_id
        return self._service.files().list(**kwargs).execute().get("files", [])

    def _resolve_folder(self, relative_path: str, *, create: bool = False) -> str:
        """Return the Drive folder ID for *relative_path*, creating intermediates if needed."""
        if relative_path in self._folder_cache:
            return self._folder_cache[relative_path]

        parts = Path(relative_path).parts
        current_id = self._folder_cache[""]
        built = ""

        for part in parts:
            built = str(Path(built) / part) if built else part
            if built in self._folder_cache:
                current_id = self._folder_cache[built]
                continue

            # Search for existing folder
            q = (
                f"'{current_id}' in parents and name = '{part}' "
                f"and mimeType = 'application/vnd.google-apps.folder' "
                f"and trashed = false"
            )
            matches = self._list_files(q)

            if matches:
                current_id = matches[0]["id"]
            elif create:
                meta = {
                    "name": part,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [current_id],
                }
                folder = self._service.files().create(
                    body=meta, fields="id", supportsAllDrives=True,
                ).execute()
                current_id = folder["id"]
            else:
                return ""

            self._folder_cache[built] = current_id

        return current_id

    def _api_upload(self, local_path: str, relative_path: str) -> None:
        """Upload a file or directory tree to Drive."""
        from googleapiclient.http import MediaFileUpload

        if os.path.isfile(local_path):
            parent_rel = str(Path(relative_path).parent)
            if parent_rel == ".":
                parent_rel = ""
            parent_id = self._resolve_folder(parent_rel, create=True)
            name = Path(relative_path).name

            # Check if file already exists → update; else create
            q = f"'{parent_id}' in parents and name = '{name}' and trashed = false"
            existing = self._list_files(q)
            media = MediaFileUpload(local_path, resumable=True)

            if existing:
                file_id = existing[0]["id"]
                self._service.files().update(
                    fileId=file_id, media_body=media, supportsAllDrives=True,
                ).execute()
            else:
                meta = {"name": name, "parents": [parent_id]}
                self._service.files().create(
                    body=meta, media_body=media, fields="id", supportsAllDrives=True,
                ).execute()
        else:
            # Directory — walk and upload each file
            local_root = Path(local_path)
            for file_path in sorted(local_root.rglob("*")):
                if file_path.is_dir():
                    continue
                rel = str(Path(relative_path) / file_path.relative_to(local_root))
                self._api_upload(str(file_path), rel)

    def _api_download(self, relative_path: str, local_path: str) -> None:
        """Download a file or directory tree from Drive."""
        # Determine if it's a folder or file
        parent_rel = str(Path(relative_path).parent)
        if parent_rel == ".":
            parent_rel = ""
        parent_id = self._resolve_folder(parent_rel)
        if not parent_id:
            logger.info("restore: folder %s not found on Drive, skipping", parent_rel)
            return

        name = Path(relative_path).name
        q = f"'{parent_id}' in parents and name = '{name}' and trashed = false"
        matches = self._list_files(q, fields="files(id, mimeType)")
        if not matches:
            logger.info("restore: %s not found on Drive, skipping", relative_path)
            return

        item = matches[0]
        if item["mimeType"] == "application/vnd.google-apps.folder":
            self._download_folder(item["id"], relative_path, local_path)
        else:
            self._download_file(item["id"], local_path)

    def _download_file(self, file_id: str, local_path: str) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def _download_folder(self, folder_id: str, relative_path: str, local_path: str) -> None:
        os.makedirs(local_path, exist_ok=True)
        q = f"'{folder_id}' in parents and trashed = false"
        items = self._list_files(q, fields="files(id, name, mimeType)")
        for item in items:
            child_local = os.path.join(local_path, item["name"])
            child_rel = str(Path(relative_path) / item["name"])
            if item["mimeType"] == "application/vnd.google-apps.folder":
                self._download_folder(item["id"], child_rel, child_local)
            else:
                self._download_file(item["id"], child_local)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_drive_service(credentials_path: str):
    """Build an authorized Drive v3 service from a service-account JSON key."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _safe_copy2(src: str, dst: str, **kwargs) -> None:
    """copy2 that silently skips when src and dst are the same file (Drive FUSE)."""
    try:
        shutil.copy2(src, dst, **kwargs)
    except shutil.SameFileError:
        pass


def _copy_local(src: str, dst: str) -> None:
    """Copy *src* to *dst*, handling both files and directories."""
    if os.path.isdir(src):
        if os.path.exists(dst):
            # Merge into existing directory
            shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_safe_copy2)
        else:
            shutil.copytree(src, dst, copy_function=_safe_copy2)
    else:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        _safe_copy2(src, dst)
