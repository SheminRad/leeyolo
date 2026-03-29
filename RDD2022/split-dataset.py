import os
import random
import shutil


def main():
    print("1. Setting up paths...")
    # Source paths (Where everything is right now)
    drone = "Czech"
    train_images_dir = "%s/train/images" % drone
    train_labels_dir = "%s/train/labels" % drone

    # Destination paths (Where we are moving the 20% validation data)
    val_images_dir = "%s/val/images" % drone
    val_labels_dir = "%s/val/labels" % drone

    # Create the new val directories
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    print("2. Shuffling and splitting data...")
    # Get all .jpg images
    all_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(all_images)

    # Calculate 20% of the data
    split_index = int(len(all_images) * 0.2)
    val_images = all_images[:split_index]

    print(f"Moving {len(val_images)} images and labels to the Validation folder...")

    # Move the files
    for img_name in val_images:
        # Move the image
        shutil.move(os.path.join(train_images_dir, img_name),
                    os.path.join(val_images_dir, img_name))

        # Move the corresponding .txt label
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(train_labels_dir, label_name)

        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_labels_dir, label_name))

    print("✅ Dataset successfully split! 80% Train / 20% Val.")


if __name__ == '__main__':
    main()