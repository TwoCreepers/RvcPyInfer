from ..error.InferEnvError import InferEnvError
from ..error.ProviderParseError import ProviderParseError
from .OrtProviders import OrtProviders
from .OvProviders import OvProviders


class InferProviders:
    providers: OvProviders | OrtProviders

    def __init__(self, providers: OvProviders | OrtProviders) -> None:
        self.providers = providers

    def __repr__(self) -> str:
        return self.providers.__repr__()

    @property
    def is_ov(self) -> bool:
        return isinstance(self.providers, OvProviders)

    @property
    def is_ort(self) -> bool:
        return isinstance(self.providers, OrtProviders)

    @staticmethod
    def default() -> "InferProviders":
        from ..infer_env import HAS_OPENVINO, HAS_ORT

        if HAS_ORT:
            return InferProviders(OrtProviders.default())
        elif HAS_OPENVINO:
            return InferProviders(OvProviders.default())
        else:
            raise InferEnvError("没有任意一个支持的推理引擎，请至少安装一个可选模块")

    @staticmethod
    def parse(value: str) -> "InferProviders":
        if not value or value == "default" or value == "default:":
            return InferProviders.default()

        i = value.find(':')
        if i == -1 or i == len(value) - 1:
            raise ProviderParseError("错误的字符串")
        type = value[0:i]
        if type == "ov" or type == "openvino":
            return InferProviders(OvProviders.parse(value[i+1:]))
        elif type == "ort" or type == "onnxruntime":
            return InferProviders(OrtProviders.parse(value[i+1:]))
        else:
            raise ProviderParseError(f"未知的提供者类型: {type}")
