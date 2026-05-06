
import os
import tempfile


url_generators = {
    # raw GitHub image. ~9 kB
    "test": "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png",
    # video: HPX source code, 6.7 MB
    "small": "https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/1.4.0.zip",
    # resnet model from pytorch. 98M
    "large": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}

_TEST_OBJECT_NAME = "uploader-test-payload.bin"
_TEST_OBJECT_BYTES = b"SEBS_UPLOADER_TEST_PAYLOAD\n" * 512

def buckets_count():
    return (1, 1)


def _create_test_payload() -> str:
    """Create a deterministic local payload for stable calibration runs."""
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    try:
        handle.write(_TEST_OBJECT_BYTES)
        return handle.name
    finally:
        handle.close()

def generate_input(data_dir, size, benchmarks_bucket, input_buckets, output_buckets, upload_func, nosql_func):
    input_config = {"object": {}, "bucket": {}}
    input_config["object"]["url"] = url_generators[size]
    input_config["bucket"]["bucket"] = benchmarks_bucket
    input_config["bucket"]["output"] = output_buckets[0]

    if size == "test" and upload_func is not None and input_buckets:
        payload_path = _create_test_payload()
        try:
            upload_func(0, _TEST_OBJECT_NAME, payload_path)
        finally:
            os.remove(payload_path)
        input_config["object"]["key"] = _TEST_OBJECT_NAME
        input_config["bucket"]["input"] = input_buckets[0]

    return input_config
