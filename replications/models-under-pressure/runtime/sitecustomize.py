import os
from os import PathLike
from pathlib import Path

import torch


_torch_load = torch.load


def _load_pathlike_with_mmap(file, *args, **kwargs):
    if kwargs.get("mmap") and isinstance(file, PathLike):
        file = str(file)
    return _torch_load(file, *args, **kwargs)


torch.load = _load_pathlike_with_mmap


if os.environ.get("MUP_LOCAL_ACTIVATIONS") == "1":
    from models_under_pressure import r2

    r2.download_file = lambda _bucket, _key, path: Path(path).exists()
    r2.upload_file = lambda _bucket, _key, _path: True
    r2.list_bucket_files = lambda _bucket: []
    r2.delete_file = lambda _bucket, _key: True
