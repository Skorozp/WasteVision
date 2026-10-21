from pathlib import Path
import zipfile
import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "raw"

raw_dir.mkdir(parents=True, exist_ok=True)

zip_path = raw_dir / "realwaste.zip"

if not zip_path.exists():
    print("Downloading RealWaste...")
    response = requests.get(
        DATASET_URL,
        stream=True
    )
    response.raise_for_status()

    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file.write(chunk)
else:
    print("Archive already downloaded.")

extract_marker = raw_dir / ".realwaste_extracted"

if not extract_marker.exists():
    print("Extracting RealWaste...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)
    extract_marker.touch()

else:
    print("Dataset already extracted.")