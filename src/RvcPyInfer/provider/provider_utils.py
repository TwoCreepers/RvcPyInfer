from .InferProviders import InferProviders
from .provider_type_alist import ProvidersLike


def infer_providers(providers_like: ProvidersLike) -> InferProviders:
    if isinstance(providers_like, InferProviders):
        return providers_like
    else:
        return InferProviders(providers_like)