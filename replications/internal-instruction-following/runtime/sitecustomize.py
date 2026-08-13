from __future__ import annotations

import os

import huggingface_hub
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def patch(loader) -> None:
    original = loader.from_pretrained

    def from_pretrained(name, *args, **kwargs):
        token = kwargs.pop("use_auth_toke", None)
        token = (
            kwargs.pop("use_auth_token", None) or token or os.environ.get("HF_TOKEN")
        )
        if token:
            kwargs.setdefault("token", token)
        revision = os.environ.get("GI_MODEL_REVISION")
        if revision:
            kwargs.setdefault("revision", revision)
        return original(name, *args, **kwargs)

    loader.from_pretrained = staticmethod(from_pretrained)


for target in (AutoConfig, AutoModelForCausalLM, AutoTokenizer):
    patch(target)


original_login = huggingface_hub.login


def login(token=None, *args, **kwargs):
    return original_login(token or os.environ.get("HF_TOKEN"), *args, **kwargs)


huggingface_hub.login = login
