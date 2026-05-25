import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess(root_dir):
    # Look in processed /train/cat first then fall back to raw/Cat
    sample_path = None
    print("STEP 3: Image Preprocessing")
    search_dirs = [
        Path(f"{root_dir}/data/processed/train/cat"),
        Path(f"{root_dir}/data/processed/train/dog"),
        Path(f"{root_dir}/data/raw/Cat"),
        Path(f"{root_dir}/data/raw/Dog"),
    ]
    for d in search_dirs:
        if d.exists():
            files = list(d.iterdir())
            if files:
                sample_path = files[0]
                break
    if sample_path is None:
        print("Image not Found")
        sys.exit(1)
    print(f"Using sample image: {sample_path}")

    raw_img = Image.open(sample_path).convert("RGB")
    raw_arr = np.array(raw_img, dtype=np.float32)
    print(f"""
    RAW IMAGE (straight from disk):
      Size   : {raw_img.size[0]} x {raw_img.size[1]} pixels
      Shape  : {raw_arr.shape}   (height, width, RGB channels)
      Dtype  : {raw_arr.dtype}
      Min px : {raw_arr.min():.0f}
      Max px : {raw_arr.max():.0f}
      Mean px: {raw_arr.mean():.1f}
    """)
    #Resize
    IMG_SIZE = (224, 224)

    img_tensor = tf.constant(raw_arr)  # convert to tensor
    img_resized = tf.image.resize(img_tensor, IMG_SIZE)  # resize to 224x224
    print(f"""AFTER RESIZE to {IMG_SIZE}:
      Shape  : {img_resized.shape}
      Min px : {img_resized.numpy().min():.0f}
      Max px : {img_resized.numpy().max():.0f}
      (pixel values unchanged — only the dimensions changed)
    """)

    # Normalize
    img_normalised = tf.cast(img_resized, tf.float32) / 255.0

    print(f"""AFTER NORMALISATION (÷ 255):
      Shape  : {img_normalised.shape}
      Dtype  : {img_normalised.dtype}
      Min px : {img_normalised.numpy().min():.4f}
      Max px : {img_normalised.numpy().max():.4f}
      Mean px: {img_normalised.numpy().mean():.4f}
      (same image, values now between 0.0 and 1.0)
    """)

    print("Example pixel transformation (top-left corner, first 5 pixels):")
    print(f"  Before: {img_resized.numpy()[0, :5, 0].astype(int).tolist()}")
    print(f"  After : {[round(v, 3) for v in img_normalised.numpy()[0, :5, 0].tolist()]}")
    return sample_path


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    print(root_dir)
    preprocess(root_dir)
