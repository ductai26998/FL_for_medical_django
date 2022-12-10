# Use when aws is error
import os
import pickle

STORAGE_FOLDER = "src/storage/"


def upload_params_to_storage(file, folder: str, file_name):
    if not file_name:
        file_name = file.name

    object_name = STORAGE_FOLDER + folder + "/" + file_name
    os.makedirs(os.path.dirname(object_name), exist_ok=True)
    with open(object_name, "wb") as f:
        f.write(file)

    return object_name


def read_params_from_storage(object_name):
    params = None
    with open(object_name, "rb") as f:
        file_content = f.read()
    params = pickle.loads(file_content)
    return params
