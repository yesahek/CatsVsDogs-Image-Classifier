import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
from src.preprocess import preprocess



def augmentation(sample_path ,root_path):
    print("  STEP 4: Data Augmentation")

    IMG_SIZE = (224, 224)
    img = tf.keras.utils.load_img(str(sample_path), target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    batch = tf.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)

    augmentations = {
        "RandomFlip (horizontal)": layers.RandomFlip("horizontal"),
        "RandomRotation (±20°)": layers.RandomRotation(0.20),
        "RandomZoom (±15%)": layers.RandomZoom(0.15),
        "RandomTranslation (±10%)": layers.RandomTranslation(0.10, 0.10),
    }
    print("Each augmentation and what it does:")
    print()
    for name, aug_layer in augmentations.items():
        augmented = aug_layer(batch, training=True)[0].numpy()
        augmented = np.clip(augmented, 0, 1)

        # Compare original vs augmented pixel stats
        diff = np.abs(arr - augmented).mean()

        descriptions = {
            "RandomFlip (horizontal)": "Mirrors image left-right. A cat facing left = a cat facing right.",
            "RandomRotation (±20°)": "Rotates up to ±20°. Cats are not always perfectly upright.",
            "RandomZoom (±15%)": "Zooms in or out. Cats appear at different distances from camera.",
            "RandomTranslation (±10%)": "Shifts image. The subject is not always perfectly centred.",
        }

        print(f"  {name}")
        print(f"    What it does : {descriptions[name]}")
        print(f"    Avg px change: {diff:.4f}  (0 = identical, 1 = completely different)")
        print()

        # ── Full pipeline (all augmentations together) ─
        print("-" * 55)
        print("Full augmentation pipeline (all 4 applied together):")
        print("""
          data_augmentation = tf.keras.Sequential([
              layers.RandomFlip("horizontal"),
              layers.RandomRotation(0.20),
              layers.RandomZoom(0.15),
              layers.RandomTranslation(0.10, 0.10),
          ])
        """)

        full_pipeline = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.20),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.10, 0.10),
        ])

        print("Applying full pipeline 5 times to the same image:")
        for i in range(5):
            aug = full_pipeline(batch, training=True)[0].numpy()
            aug = np.clip(aug, 0, 1)
            diff = np.abs(arr - aug).mean()
            print(f"  Version {i + 1}: avg pixel change = {diff:.4f}")

        print()

        # ── Save visualisation if matplotlib available ────────────────────────────────
        try:

            matplotlib.use("Agg")

            Path("docs").mkdir(exist_ok=True)
            fig, axes = plt.subplots(2, 5, figsize=(16, 7))
            fig.suptitle(
                "Data Augmentation — Same Image, 9 Different Views\n"
                "(Training set only — val/test are never augmented)",
                fontsize=13, fontweight="bold"
            )

            axes[0, 0].imshow(arr)
            axes[0, 0].set_title("Original", fontsize=10, fontweight="bold", color="green")
            axes[0, 0].axis("off")

            for i, ax in enumerate(axes.flat[1:]):
                aug = full_pipeline(batch, training=True)[0].numpy()
                ax.imshow(np.clip(aug, 0, 1))
                ax.set_title(f"Aug #{i + 1}", fontsize=9)
                ax.axis("off")

            plt.tight_layout()
            save_path = root_path / "docs/augmentation_demo.png"
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"✅ Visualisation saved → {save_path}")
        except ImportError:
            print("(Install matplotlib to save visualisation: pip install matplotlib)")

        print()
        print("Augmentation demo done!")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent

    #sample_path = root_dir / "data/processed/train/cat/cat_0.jpg"
    sample_path = preprocess(root_dir)
    augmentation(sample_path, root_dir)