import os
import shutil
import re

def clean_plate_number(plate):
    """Clean the plate number by removing hyphens and spaces."""
    return re.sub(r'[^A-Za-z0-9]', '', plate)

def rearrange_data():
    # Define paths
    base_dir = "data/license_recognition/inputs"
    new_base_dir = "dataset"

    # Old directories
    old_train_images = os.path.join(base_dir, "train", "images")
    old_train_labels = os.path.join(base_dir, "train", "labels")
    old_val_images = os.path.join(base_dir, "val", "images")
    old_val_labels = os.path.join(base_dir, "val", "labels")

    # New directories
    new_train_dir = os.path.join(new_base_dir, "train")
    new_val_dir = os.path.join(new_base_dir, "val")

    # Create new directories
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_val_dir, exist_ok=True)

    def process_split(images_dir, labels_dir, new_dir):
        # Get list of label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        for label_file in label_files:
            # Get the number (without extension)
            num = os.path.splitext(label_file)[0]
            # Read the plate number
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                plate = f.read().strip()
            # Clean the plate number
            clean_plate = clean_plate_number(plate)
            # Source image
            src_image = os.path.join(images_dir, f"{num}.jpg")
            # Destination image
            dst_image = os.path.join(new_dir, f"{clean_plate}.jpg")
            # Copy and rename
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            else:
                print(f"Warning: Image {src_image} not found")

    # Process train
    process_split(old_train_images, old_train_labels, new_train_dir)
    # Process val
    process_split(old_val_images, old_val_labels, new_val_dir)

    # Remove old structure
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"Removed old directory: {base_dir}")

if __name__ == "__main__":
    rearrange_data()