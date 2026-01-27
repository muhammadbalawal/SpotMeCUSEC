"""
Creates a mapping of filename -> Google Drive file ID for direct linking.
"""

import os
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Config
FOLDER_ID = "1lFLblGFIMAGiD4oK_Jt7N2M-ZSV4GcTa"
SERVICE_ACCOUNT_FILE = "service_account.json"
OUTPUT_FILE = "data/drive_mapping.json"

# Auth
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
service = build("drive", "v3", credentials=creds)


def list_all_files(folder_id):
    """List all files in a Drive folder"""
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"

    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return files


def main():
    print("Fetching files from Google Drive...")
    files = list_all_files(FOLDER_ID)
    print(f"Found {len(files)} files")

    # Create mapping: filename -> file_id (only JPG files)
    mapping = {}
    for f in files:
        name = f["name"]
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            mapping[name] = f["id"]

    print(f"Filtered to {len(mapping)} image files (JPG/PNG/WebP only)")

    # Save mapping
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Mapping saved to {OUTPUT_FILE}")
    print(f"Total files mapped: {len(mapping)}")


if __name__ == "__main__":
    main()
