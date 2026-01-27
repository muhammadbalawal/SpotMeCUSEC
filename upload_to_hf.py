from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='.',
    repo_id='bscgi/spotme-cusec',
    repo_type='space',
    ignore_patterns=['venv/*', '.git/*', '__pycache__/*', 'data/cache/*', 'data/photos/*', 'upload_to_hf.py']
)
print("Upload complete!")
