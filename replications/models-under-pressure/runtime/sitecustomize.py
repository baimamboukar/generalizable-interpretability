from os import PathLike

import torch


_torch_load = torch.load


def _load_pathlike_with_mmap(file, *args, **kwargs):
    if kwargs.get("mmap") and isinstance(file, PathLike):
        file = str(file)
    return _torch_load(file, *args, **kwargs)


torch.load = _load_pathlike_with_mmap
