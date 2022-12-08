#  This file to extract data to classification folder to train

from imutils import paths
import csv


def main():
    create_annotations(dataset_type="train")
    create_annotations(dataset_type="test")


def write_to_csv(path: str, rows: list):
    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def create_annotations(dataset_type: str):
    """
    Params:
    - dataset_type: train or test
    """
    img_dir = f"train/data/custom/{dataset_type}/images"
    csv_path = f"train/data/custom/{dataset_type}/annotations.csv"
    image_paths = set(paths.list_images(img_dir))
    rows = []
    count = 0
    for img_path in image_paths:
        pathology = img_path.split("/")[-2]
        label = None
        if pathology == "benign":
            label = 0
        elif pathology == "benignwithoutcallback":
            label = 1
        elif pathology == "malignant":
            label = 2
        else:
            pass
        if label is not None:
            count += 1
            rows.append((img_path, label))
        else:
            print("%s is invalid pathology" % pathology)
    print("count", count)
    write_to_csv(path=csv_path, rows=rows)


if __name__ == "__main__":
    main()
