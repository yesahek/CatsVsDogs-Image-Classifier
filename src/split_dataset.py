import os
import random
import shutil
from pathlib import Path



def split_dataset(RAW_DIR, PROCESSED_DIR, TEST_SPLIT, VAL_SPLIT, SEED):
    print("  STEP 2: Splitting the Dataset")
    print(f"""
    Raw data folder  : {RAW_DIR}
    Processed folder : {PROCESSED_DIR}
    Test split       : {TEST_SPLIT * 100:.0f}%  (carved out FIRST — never seen during training)
    Val split        : {VAL_SPLIT * 100:.0f}%  (of remaining data, used to monitor training)
    Train            : the rest (~{(1 - TEST_SPLIT) * (1 - VAL_SPLIT) * 100:.0f}%)
    Seed             : {SEED}
    """)

    RAW_FOLDER_MAP = {
        "Cat":"cat",
        "Dog":"dog"
    }

    # checking raw folder exists -------
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print(f" Raw folder not found: {RAW_DIR}")

    random.seed(42)
    stats = {"train":{},"val":{},"test":{}}
    for raw_folder, label in RAW_FOLDER_MAP.items():
        src_dir = raw_path / raw_folder

        if not src_dir.exists():
            print(f"Folder not found: {src_dir}")
            exit(1)

        # Collect all image files, sorted numerically
        all_files = sorted(
            [p for p in src_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )
        print(f"Found {len(all_files):,} images in {src_dir}")

        #shuffle reproducibly
        random.shuffle(all_files)
        n_total = len(all_files)
        n_test = int(n_total * TEST_SPLIT)
        n_remaining = n_total - n_test
        n_val = int(n_remaining * VAL_SPLIT)

        splits = {
            "test": all_files[:n_test],
            "val": all_files[n_test: n_test + n_val],
            "train": all_files[n_test + n_val:],
        }
        # Copy into processed/ with renamed files
        for split_name, files in splits.items():
            dest_dir = Path(PROCESSED_DIR) / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                # Rename: 1.jpg → cat_1.jpg  (avoids cat/dog filename collisions)
                new_name = f"{label}_{f.name}"
                shutil.copy2(f, dest_dir / new_name)

            stats[split_name][label] = len(files)

        print(f"  {raw_folder} → train: {stats['train'][label]:>5} | "
              f"val: {stats['val'][label]:>4} | "
              f"test: {stats['test'][label]:>4}")

    # ── Print summary ─────────────────────────────────────────────────────────────
    total = sum(v for split in stats.values() for v in split.values())

    print(f"""
        ✅ Split complete! {total:,} images organised into:

           data/processed/
           ├── train/cat/  ({stats['train']['cat']:,})    ├── train/dog/  ({stats['train']['dog']:,})
           ├── val/cat/    ({stats['val']['cat']:,})      ├── val/dog/    ({stats['val']['dog']:,})
           └── test/cat/   ({stats['test']['cat']:,})     └── test/dog/   ({stats['test']['dog']:,})
    """)




if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    RAW_DIR       = root_dir /"data"/"raw"/ "PetImages"        # where your Cat/ and Dog/ folders live
    PROCESSED_DIR = root_dir /"data/processed"  # where the split dataset will be saved
    TEST_SPLIT    = 0.20              # 20% → test set  (held out first)
    VAL_SPLIT     = 0.20              # 20% of remaining → val set
    SEED          = 42
    split_dataset(RAW_DIR, PROCESSED_DIR, TEST_SPLIT, VAL_SPLIT, SEED)