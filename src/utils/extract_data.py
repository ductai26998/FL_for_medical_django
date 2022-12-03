import pandas as pd
from imutils import paths
import csv


def main():
    create_annotations(dataset_type="train")
    create_annotations(dataset_type="test")


#     parent_dir = "train/breast_cancer_images"
#     image_dir = f"{parent_dir}/jpeg"
#     dicom_data = pd.read_csv(f"{parent_dir}/csv/dicom_info.csv")
#     # cropped images
#     cropped_images = dicom_data[
#         dicom_data.SeriesDescription == "cropped images"
#     ].image_path
#     cropped_images = cropped_images.apply(
#         lambda x: x.replace("CBIS-DDSM/jpeg", image_dir)
#     )
#     # for file in cropped_images[0:10000]:
#     #     if file[-9] not in ["1","2"]:
#     #         print(file)
#     # full mammogram images
#     full_mammogram_images = dicom_data[
#         dicom_data.SeriesDescription == "full mammogram images"
#     ].image_path
#     full_mammogram_images = full_mammogram_images.apply(
#         lambda x: x.replace("CBIS-DDSM/jpeg", image_dir)
#     )
#     images = list(cropped_images) + list(full_mammogram_images)


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