import os
import subprocess
from pathlib import Path



def load_kaggle_json(config_dir):
    PROJECT_KAGGLE_DIR = Path(config_dir)
    KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)

    TARGET_JSON_PATH = PROJECT_KAGGLE_DIR / "kaggle.json"

    os.environ["KAGGLE_CONFIG_DIR"] = str(PROJECT_KAGGLE_DIR.resolve())

    if TARGET_JSON_PATH.exists():
        os.chmod(TARGET_JSON_PATH, 0o600)
        print(f"✅ kaggle.json is configured and secured at: {TARGET_JSON_PATH}")
        return True
    else:
        print(f"❌ kaggle.json NOT found at: {TARGET_JSON_PATH}")
        print("👉 Action Required: Please download your token from Kaggle and drop it into that folder!")
        return False

def is_dataset_downloaded(download_dir):
    #verifying if the 'Cat' and 'Dog' folders exist and are not empty.

    row_path = Path(download_dir)
    cat_dir = row_path /"PetImages" /"Cat"
    dog_dir = row_path /"PetImages" /"Dog"

    if cat_dir.exists() and dog_dir.exists():
        has_cast = any(cat_dir.iterdir())
        has_dogs = any(dog_dir.iterdir())

        if has_dogs and has_cast:
            return True
    return False

def download_dataset(download_path, dataset_link):
    print("Starting the kaggle dataset download.....")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", str(dataset_link),
        "-p", str(download_path),
        "--unzip"
    ], check=True)



def check_downloads(download_dir):
    # Check what was downloaded with corrected local indentation
    print("\n📂 Downloaded files:")
    raw_path = Path(download_dir).resolve()

    for p in sorted(raw_path.rglob("*"))[:20]:
        # Calculate indentation relative to your target raw data folder root
        depth = len(p.relative_to(raw_path).parts)
        indent = "  " * depth
        print(f"{indent}{'📁' if p.is_dir() else '🖼️ '} {p.name}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent

    RAW_DATA_DIR = f"{root_dir}/data/raw/"
    CONFIG_DIR = root_dir /"config/"
    DATASET_LINK = "shaunthesheep/microsoft-catsvsdogs-dataset"

    if load_kaggle_json(CONFIG_DIR):
        print("Checking if the dataset is already downloaded...")
        if is_dataset_downloaded(RAW_DATA_DIR):
            print("Dataset already exists locally! Skipping downloading step.")
        else:
            print("Dataset missing or incomplete")
            download_dataset(RAW_DATA_DIR,DATASET_LINK)
        check_downloads(RAW_DATA_DIR)
    else:
        print("\n Execution stopped: Fix kaggle.json file placement first.")







