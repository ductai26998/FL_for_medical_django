#  This file to extract data from an folder to data folder to train

import glob
import shutil
from tqdm import tqdm

TRAIN_CANCER_FOLDER_PATH = "src/train/data/train/1"
TRAIN_NO_CANCER_FOLDER_PATH = "src/train/data/train/0"
VALIDATION_CANCER_FOLDER_PATH = "src/train/data/validation/1"
VALIDATION_NO_CANCER_FOLDER_PATH = "src/train/data/validation/0"
TEST_CANCER_FOLDER_PATH = "src/train/data/test/1"
TEST_NO_CANCER_FOLDER_PATH = "src/train/data/test/0"


def main():
    """
    Group the histopathology images to dataset to train with ImageDataGenerator class
    """
    img_path = "src/train/breast_cancer_images/**/**/*.png"
    breast_img = glob.glob(img_path, recursive=True)
    non_can_img = []
    can_img = []
    for img in breast_img:
        if img[-5] == "0":
            non_can_img.append(img)

        elif img[-5] == "1":
            can_img.append(img)

    total_non_can = len(non_can_img)
    total_can = len(can_img)

    total_non_can_train = round(total_non_can / 10 * 7)
    total_can_train = round(total_can / 10 * 2)

    total_non_can_val = round(total_non_can / 10 * 7)
    total_can_val = round(total_can / 10 * 2)

    print("Copying non images")
    for i in tqdm(range(0, total_non_can)):
        if i < total_non_can_train:
            shutil.copy2(non_can_img[i], TRAIN_NO_CANCER_FOLDER_PATH)
        elif i < total_non_can_train + total_non_can_val:
            shutil.copy2(non_can_img[i], VALIDATION_NO_CANCER_FOLDER_PATH)
        else:
            shutil.copy2(non_can_img[i], TEST_NO_CANCER_FOLDER_PATH)

    print("Copying cancer images")
    for i in tqdm(range(0, total_can)):
        if i < total_can_train:
            shutil.copy2(can_img[i], TRAIN_CANCER_FOLDER_PATH)
        elif i < total_can_train + total_can_val:
            shutil.copy2(can_img[i], VALIDATION_CANCER_FOLDER_PATH)
        else:
            shutil.copy2(can_img[i], TEST_CANCER_FOLDER_PATH)


if __name__ == "__main__":
    main()
