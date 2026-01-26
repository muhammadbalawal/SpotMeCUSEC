kkimport os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ----------------------------
# CONFIG
# ----------------------------
FOLDER_ID = "1lFLblGFIMAGiD4oK_Jt7N2M-ZSV4GcTa"
OUTPUT_DIR = "data/photos"
SERVICE_ACCOUNT_FILE = "service_account.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# AUTHENTICATION
# ----------------------------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build("drive", "drive", credentials=creds)

# ----------------------------
# LIST FILES IN FOLDER
# ----------------------------
def list_files(folder_id):
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
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break
    return files

# ----------------------------
# DOWNLOAD SINGLE FILE
# ----------------------------
def download_file(file):
    file_id = file["id"]
    file_name = file["name"]
    output_path = os.path.join(OUTPUT_DIR, file_name)

    if os.path.exists(output_path):
        print(f"Skipping {file_name}, already downloaded.")
        return

    request = service.files().get_media(fileId=file_id)
    tmp_path = output_path + ".tmp"

    with io.FileIO(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"{file_name}: {int(status.progress() * 100)}% downloaded", end="\r")

    os.rename(tmp_path, output_path)
    print(f"{file_name}: Download complete!{' ' * 20}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    files = list_files(FOLDER_ID)
    print(f"Found {len(files)} images in the folder.\n")

    for i, file in enumerate(files, 1):
        print(f"Downloading {i}/{len(files)}: {file['name']}")
        download_file(file)

    print("\nAll downloads complete!")

