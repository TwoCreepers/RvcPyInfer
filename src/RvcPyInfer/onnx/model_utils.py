from pathlib import Path

from ..provider.InferProviders import InferProviders
from ..provider.OrtProviders import OrtProviders
from .InferModel import InferModel
from .OrtModel import OrtModel
from .OvModel import OvModel


def build_model(model: Path, provider: InferProviders) -> InferModel:
    if isinstance(provider.providers, OrtProviders):
        return OrtModel(model, provider.providers)
    else:
        return OvModel(model, provider.providers)