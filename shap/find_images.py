import os


def get_image_lists():
    base_dir = "/home/shemin/PycharmProjects/leeyolo/RDD2022/China_Drone/train"
    labels_dir = os.path.join(base_dir, "labels")
    images_dir = os.path.join(base_dir, "images")

    # Initialize the specific lists requested
    longitudinal_crack = []  # Class 0
    transverse_crack = []  # Class 1
    alligator_crack = []  # Class 2
    pothole = []  # Class 3

    lists = [longitudinal_crack, transverse_crack, alligator_crack, pothole]
    max_per_class = 5

    if not os.path.exists(labels_dir):
        print(f"Directory not found: {labels_dir}")
        return lists

    # Standard directory scanning and distribution logic applied
    for label_filename in os.listdir(labels_dir):
        if not label_filename.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_filename)

        with open(label_path, 'r') as file:
            for line in file:
                class_id = int(line.split()[0])

                if 0 <= class_id <= 3:
                    if len(lists[class_id]) < max_per_class:
                        image_filename = f"{os.path.splitext(label_filename)[0]}.jpg"
                        full_image_path = os.path.join(images_dir, image_filename)

                        # Ensure no duplicates and the image actually exists
                        if os.path.exists(full_image_path) and image_filename not in lists[class_id]:
                            lists[class_id].append(image_filename)

        # Optimization: Stop searching if all lists have hit their 5-image quota
        if all(len(lst) >= max_per_class for lst in lists):
            break

    return longitudinal_crack, transverse_crack, alligator_crack, pothole