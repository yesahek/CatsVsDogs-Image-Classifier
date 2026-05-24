from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


#finding the Dataset
def check_dataset(raw_data):
    cat_dir = raw_data/"Cat"
    dog_dir = raw_data/"Dog"

    if dog_dir.exists() and cat_dir.exists():
       has_dogs = any(dog_dir.iterdir())
       has_cats = any(cat_dir.iterdir())
       if has_dogs and has_cats:
           return True
    return False

def basic_info(raw_dir):
    cat_dir = raw_dir /"Cat"
    dog_dir = raw_dir /"Dog"
    print("Basic Info ===================")
    print(f"Cat Images: {len(list(cat_dir.iterdir()))}")
    print(f"Dog Images: {len(list(dog_dir.iterdir()))}")

    print("Understanding the Data")
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    fig.suptitle("Raw Dataset — Sample Images", fontsize=14, fontweight="bold", y=1.01)

    for row, (cls_dir, cls_name) in enumerate([(cat_dir, "Cat"), (dog_dir, "Dog")]):
        files = sorted(cls_dir.iterdir())[:6]

        for col, fpath in enumerate(files):
            img =Image.open(fpath).convert("RGB")
            axis[row, col].imshow(img)
            axis[row, col].set_title(f"{cls_name}/{fpath.name}\n{img.size[0]}×{img.size[1]}", fontsize=8),
            axis[row, col].axis("off")
    plt.tight_layout()
    plt.show()

    n_cats = len(list(cat_dir.iterdir()))
    n_dogs = len(list(dog_dir.iterdir()))
    total = n_cats + n_dogs

    print(f"Cat Images: {n_cats:,}({n_cats/total*100:.1f})%")
    print(f"Dog Images: {n_dogs:,}({n_dogs/total*100:.1f})%")

    # checking the dataset is balanced or not
    if abs(n_cats - n_dogs)/100 <0.5:
        print("dataset is balanced")
    else:
        print("dataset is imbalanced")

    #Bar chart for class distrbution
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    bars = ax.bar(["Cat", "Dog"], [n_cats, n_dogs], color=["#FEF3C7", "#DBEAFE"], edgecolor=["#F59E0B", "#3B82F6"],
                  linewidth=2)
    for bar, n in zip(bars, [n_cats, n_dogs]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{n:,}", ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Class Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(n_cats, n_dogs) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


    print("Checking Image size")
    sizes = {"Cat":[],"Dog":[]}
    for cls_name,cls_dir, in [("Cat",cat_dir),("Dog",dog_dir)]:
        sample_files = list(cls_dir.iterdir())[:100]
        for f in sample_files:
            try:
                w, h = Image.open(f).size
                sizes[cls_name].append((w,h))
            except:
                pass
    for cls_name, size_list in sizes.items():
        widths = [s[0] for s in size_list]
        heights = [s[1] for s in size_list]
        print(f"\n  {cls_name}:")
        print(f"    Width  — min: {min(widths)}, max: {max(widths)}, avg: {int(np.mean(widths))}")
        print(f"    Height — min: {min(heights)}, max: {max(heights)}, avg: {int(np.mean(heights))}")

    print()
    print("→ Images have different sizes — we MUST resize to a fixed size before training")
    print("→ We will resize everything to 224×224 (required by VGG16, ResNet50, EfficientNet)")






if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    raw_data = root_dir / "data" / "raw" / "PetImages"

    print("Checking datasets if the Cat and Dog Folder is exists...")
    if check_dataset(raw_data):
        print("Found Datasets")
    else:
        print("First download the dataset or run the data_loader.py file")

    basic_info(raw_data)
