import os
import random
import shutil


def split_country_data(base_dir, country_name):
    print(f"\n--- Processing {country_name} ---")

    # 1. Source paths
    country_path = os.path.join(base_dir, country_name)
    train_images_dir = os.path.join(country_path, "train/images")
    train_labels_dir = os.path.join(country_path, "train/labels")

    # 2. Destination paths
    val_images_dir = os.path.join(country_path, "val/images")
    val_labels_dir = os.path.join(country_path, "val/labels")

    # Create the new val directories
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 3. Get all images (handling jpg, jpeg, png safely)
    all_images = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not all_images:
        print(f"⚠️ No images found in {train_images_dir}!")
        return

    # Shuffle for a fair distribution
    random.seed(42)
    random.shuffle(all_images)

    # Calculate 20% of the data
    split_index = int(len(all_images) * 0.2)
    val_images = all_images[:split_index]

    print(f"Moving {len(val_images)} images and labels to the Validation folder...")

    # 4. Move the files
    for img_name in val_images:
        # Move the image
        shutil.move(os.path.join(train_images_dir, img_name),
                    os.path.join(val_images_dir, img_name))

        # Find and move the corresponding label (.txt)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)

        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_labels_dir, label_name))

    print(f"✅ {country_name} successfully split!")


def main():
    # The root folder shown in your screenshot
    base_dataset_dir = "clustered_JapanUS"

    # Process both folders automatically
    countries = ["Japan", "United_States"]

    for country in countries:
        split_country_data(base_dataset_dir, country)

    print("\n🚀 All datasets prepped for training!")


if __name__ == '__main__':
    main()