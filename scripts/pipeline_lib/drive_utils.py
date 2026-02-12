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

        if mode == "service_account":
            if not credentials_path:
                raise ValueError("credentials_path required for service_account mode")
            if not folder_id:
                raise ValueError("folder_id required for service_account mode")
            self._folder_id = folder_id
            self._service = _build_drive_service(credentials_path)
            # Cache: relative_path -> Drive folder ID
            self._folder_cache: dict[str, str] = {"": folder_id}

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
            self._api_upload(local_path, drive_relative_path)

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
            self._resolve_folder(drive_relative_path, create=True)

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
            resp = self._service.files().list(q=q, fields="files(id)", spaces="drive").execute()
            matches = resp.get("files", [])

            if matches:
                current_id = matches[0]["id"]
            elif create:
                meta = {
                    "name": part,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [current_id],
                }
                folder = self._service.files().create(body=meta, fields="id").execute()
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
            existing = self._service.files().list(q=q, fields="files(id)", spaces="drive").execute()
            media = MediaFileUpload(local_path, resumable=True)

            if existing.get("files"):
                file_id = existing["files"][0]["id"]
                self._service.files().update(fileId=file_id, media_body=media).execute()
            else:
                meta = {"name": name, "parents": [parent_id]}
                self._service.files().create(body=meta, media_body=media, fields="id").execute()
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
        resp = self._service.files().list(q=q, fields="files(id, mimeType)", spaces="drive").execute()
        matches = resp.get("files", [])
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
        request = self._service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def _download_folder(self, folder_id: str, relative_path: str, local_path: str) -> None:
        os.makedirs(local_path, exist_ok=True)
        q = f"'{folder_id}' in parents and trashed = false"
        resp = self._service.files().list(q=q, fields="files(id, name, mimeType)", spaces="drive").execute()
        for item in resp.get("files", []):
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


def _copy_local(src: str, dst: str) -> None:
    """Copy *src* to *dst*, handling both files and directories."""
    if os.path.isdir(src):
        if os.path.exists(dst):
            # Merge into existing directory
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copytree(src, dst)
    else:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
