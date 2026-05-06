
import datetime
import os
import shutil
import uuid

import urllib.request

from . import storage
client = storage.storage.get_instance()
_DOWNLOAD_USER_AGENT = "Mozilla/5.0 (compatible; SeBSUploader/1.0)"


def _download_to_path(url: str, download_path: str) -> None:
    """Download a remote object with an explicit user agent.

    Some public hosting endpoints reject Python's default urllib user agent and
    return HTTP 403. Setting a browser-like user agent keeps the benchmark input
    stable across providers.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        with open(download_path, "wb") as output_file:
            shutil.copyfileobj(response, output_file)


def _download_input(event: dict, download_path: str) -> str:
    """Download the uploader input from benchmark storage or a fallback URL."""
    bucket_cfg = event.get("bucket", {})
    object_cfg = event.get("object", {})
    source_bucket = bucket_cfg.get("bucket")
    input_prefix = bucket_cfg.get("input")
    object_key = object_cfg.get("key")
    if source_bucket and input_prefix and object_key:
        client.download(source_bucket, os.path.join(input_prefix, object_key), download_path)
        return f"storage://{source_bucket}/{os.path.join(input_prefix, object_key)}"

    url = object_cfg.get("url")
    if not url:
        raise ValueError("Uploader input requires either object.key or object.url.")
    _download_to_path(url, download_path)
    return url


def handler(event):

    bucket = event.get('bucket').get('bucket')
    output_prefix = event.get('bucket').get('output')
    source = event.get("object", {}).get("key") or event.get("object", {}).get("url")
    name = os.path.basename(source)
    download_path = "/tmp/{}_{}".format(uuid.uuid4().hex, name)

    process_begin = datetime.datetime.now()
    source_reference = _download_input(event, download_path)
    size = os.path.getsize(download_path)
    process_end = datetime.datetime.now()

    upload_begin = datetime.datetime.now()
    key_name = client.upload(bucket, os.path.join(output_prefix, name), download_path)
    upload_end = datetime.datetime.now()
    os.remove(download_path)

    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
    upload_time = (upload_end - upload_begin) / datetime.timedelta(microseconds=1)
    return {
            'result': {
                'bucket': bucket,
                'url': source_reference,
                'key': key_name
            },
            'measurement': {
                'download_time': 0,
                'download_size': 0,
                'upload_time': upload_time,
                'upload_size': size,
                'compute_time': process_time
            }
    }
