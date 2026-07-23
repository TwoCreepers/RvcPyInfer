from .InferProviders import InferProviders
from .OrtProviders import OrtProviders
from .OvProviders import OvProviders

type ProvidersLike = InferProviders | OvProviders | OrtProviders