import boto3
from django.conf import settings
import pickle


def _s3_initialize():
    session = boto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )
    s3 = session.resource("s3")
    return s3


def upload_file_to_s3(
    file, bucket_name=settings.AWS_STORAGE_BUCKET_NAME, object_name=None
):
    """Upload a file to an S3 bucket"""

    s3 = _s3_initialize()
    s3.Bucket(bucket_name).put_object(
        ACL="public-read", Key=object_name, Body=file)
    url = "https://%s.s3.amazonaws.com/%s" % (bucket_name, object_name)
    return url


def upload_params_to_s3(file, folder: str):
    object_name = "%s/%s/%s" % (settings.FOLDER, folder, file.name)
    upload_file_to_s3(file=file, object_name=object_name)
    return object_name


def read_params_from_s3(object_name, bucket_name=settings.AWS_STORAGE_BUCKET_NAME):
    s3 = _s3_initialize()
    params = pickle.loads(s3.Bucket(bucket_name).Object(
        object_name).get()['Body'].read())
    return params
