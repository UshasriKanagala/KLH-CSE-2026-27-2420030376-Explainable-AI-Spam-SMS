from pathlib import Path
import urllib.request
import zipfile

DATA_DIR = Path(__file__).parent
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
EXTRACT_DIR = DATA_DIR / "sms_data"

print("Downloading SMS Spam Collection...")

urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

print("Download completed.")

print("Extracting dataset...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extraction completed.")
print(f"Dataset location: {EXTRACT_DIR}")