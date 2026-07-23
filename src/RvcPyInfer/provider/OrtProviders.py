from enum import Enum, auto

from ..error.InferEnvError import InferEnvError
from ..error.ProviderParseError import ProviderParseError


class OrtProviderType(Enum):
    DML = auto()
    CUDA = auto()
    CPU = auto()

    def __str__(self) -> str:
        if self == OrtProviderType.DML:
            return "DML"
        elif self == OrtProviderType.CUDA:
            return "CUDA"
        elif self == OrtProviderType.CPU:
            return "CPU"
        else:
            assert False, "不可能到达的地方"

class OrtProviders:
    def __init__(self, devices: list[OrtProviderType]):
        self.devices = devices

    def __repr__(self) -> str:
        return f"OrtProviders(devices=[{', '.join(str(d) for d in self.devices)}])"

    def to_providers(self) -> list[str]:
        res: list[str] = []
        for provider in self.devices:
            if provider == OrtProviderType.CUDA:
                res.append("CUDAExecutionProvider")
            elif provider == OrtProviderType.DML:
                res.append("DmlExecutionProvider")
            elif provider == OrtProviderType.CPU:
                res.append("CPUExecutionProvider")
        return res

    @staticmethod
    def force_dml() -> "OrtProviders":
        available = OrtProviders.currently_available_no_none()
        if OrtProviderType.DML in available:
            return OrtProviders([OrtProviderType.DML])
        else:
            raise InferEnvError("DML 提供者不存在")

    @staticmethod
    def dml() -> "OrtProviders":
        res = OrtProviders.force_dml()
        res.devices.append(OrtProviderType.CPU)
        return res

    @staticmethod
    def force_cuda() -> "OrtProviders":
        available = OrtProviders.currently_available_no_none()
        if OrtProviderType.CUDA in available:
            return OrtProviders([OrtProviderType.CUDA])
        else:
            raise InferEnvError("CUDA 提供者不存在")

    @staticmethod
    def cuda() -> "OrtProviders":
        res = OrtProviders.force_cuda()
        res.devices.append(OrtProviderType.CPU)
        return res

    @staticmethod
    def currently_available() -> list[OrtProviderType] | None:
        from ..infer_env import HAS_ORT

        if HAS_ORT:
            import onnxruntime as ort  # pyright: ignore[reportMissingImports]
            res: list[OrtProviderType] = []
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                res.append(OrtProviderType.CUDA)
            if "DmlExecutionProvider" in available_providers:
                res.append(OrtProviderType.DML)
            if "CPUExecutionProvider" in available_providers:
                res.append(OrtProviderType.CPU)
            return res

    @staticmethod
    def currently_available_no_none() -> list[OrtProviderType]:
        available = OrtProviders.currently_available()
        if available is None:
            raise InferEnvError("OnnxRunntime 未安装")
        else:
            return available

    @staticmethod
    def default() -> "OrtProviders":
        available = OrtProviders.currently_available_no_none()
        return OrtProviders(available)

    @staticmethod
    def parse(value: str) -> "OrtProviders":
        if not value or value == "default":
            return OrtProviders.default()
        splited = value.split(' ')
        available = OrtProviders.currently_available_no_none()
        types: list[OrtProviderType] = []
        for s in splited:
            if s == "cuda":
                if OrtProviderType.CUDA in available:
                    types.append(OrtProviderType.CUDA)
                else:
                    raise InferEnvError("CUDA 提供者在当前环境不可用")
            elif s == "dml":
                if OrtProviderType.DML in available:
                    types.append(OrtProviderType.DML)
                else:
                    raise InferEnvError("DML 提供者在当前环境不可用")
            elif s == "cpu":
                types.append(OrtProviderType.CPU)
            else:
                raise ProviderParseError(f"未知的提供者: {s}")
        return OrtProviders(types)