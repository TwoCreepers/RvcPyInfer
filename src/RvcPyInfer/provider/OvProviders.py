from dataclasses import dataclass
from enum import Enum, auto

from ..error.InferEnvError import InferEnvError
from ..error.ProviderError import ProviderError
from ..error.ProviderParseError import ProviderParseError


class OvProviderType(Enum):
    GPU = auto()
    NPU = auto()
    CPU = auto()

    def __str__(self) -> str:
        if self == OvProviderType.GPU:
            return "GPU"
        elif self == OvProviderType.NPU:
            return "NPU"
        elif self == OvProviderType.CPU:
            return "CPU"
        else:
            assert False, "不可能到达的地方"

class OvProvidersFlag(Enum):
    NONE = auto()
    AUTO = auto()
    MULTI = auto()
    HETERO = auto()

@dataclass
class OvMultipleDevice:
    type: OvProviderType
    num: int

    def __str__(self) -> str:
        return f"{self.type}.{self.num}"

class OvProviders:
    _flag: OvProvidersFlag
    _devices: list[OvProviderType | OvMultipleDevice]

    def __init__(self, devices: list[OvProviderType | OvMultipleDevice], flag: OvProvidersFlag = OvProvidersFlag.NONE) -> None:
        self._devices = devices
        self._flag = flag

    def __repr__(self) -> str:
        return f"OvProviders(flag={self._flag}, devices=[{', '.join(str(d) for d in self._devices)}])"

    @property
    def flag(self) -> OvProvidersFlag | None:
        return self._flag

    @flag.setter
    def flag(self, value: OvProvidersFlag) -> None:
        self._flag = value

    def check_len(self) -> None:
        if self._flag != OvProvidersFlag.AUTO and len(self._devices) == 0:
            raise ProviderError("必须至少有一个设备")

    def check_available(self) -> None:
        available = OvProviders.currently_available_no_none()
        for d in self._devices:
            if d not in available:
                raise ProviderError(f"{d} 在该计算机上不可用")

    def check(self) -> None:
        self.check_len()
        self.check_available()

    def to_providers(self) -> str:
        self.check()

        res = ','.join(str(d) for d in self._devices)
        if self._flag == OvProvidersFlag.NONE:
            return res
        elif self._flag == OvProvidersFlag.AUTO:
            if res:
                return f"AUTO:{res}"
            else:
                return "AUTO"
        elif self._flag == OvProvidersFlag.MULTI:
            return f"MULTI:{res}"
        elif self._flag == OvProvidersFlag.HETERO:
            return f"HETERO:{res}"
        else:
            assert False, "不可能到达的地方"

    @staticmethod
    def currently_available() -> list[OvProviderType | OvMultipleDevice] | None:
        from ..infer_env import HAS_OPENVINO
        
        if HAS_OPENVINO:
            from ..ov.OVCoreSingleton import core
            available = core.available_devices
            res: list[OvProviderType | OvMultipleDevice] = []
            for a in available:
                sp = a.split('.')
                if len(sp) == 1:
                    if sp[0] == "GPU":
                        res.append(OvProviderType.GPU)
                    elif sp[0] == "NPU":
                        res.append(OvProviderType.NPU)
                    elif sp[0] == "CPU":
                        res.append(OvProviderType.CPU)
                else:
                    if sp[0] == "GPU":
                        res.append(OvMultipleDevice(OvProviderType.GPU, int(sp[1])))
                    elif sp[0] == "NPU":
                        res.append(OvMultipleDevice(OvProviderType.NPU, int(sp[1])))
                    elif sp[0] == "CPU":
                        res.append(OvMultipleDevice(OvProviderType.CPU, int(sp[1])))
            return res

    @staticmethod
    def currently_available_no_none() -> list[OvProviderType | OvMultipleDevice]:
        available = OvProviders.currently_available()
        if available is None:
            raise InferEnvError("OpenVINO 未安装")
        else:
            return available

    @staticmethod
    def default() -> "OvProviders":
        available = OvProviders.currently_available_no_none()
        if any(isinstance(a, OvMultipleDevice) and a.type == OvProviderType.GPU for a in available):
            return OvProviders([OvMultipleDevice(OvProviderType.GPU, 0), OvProviderType.CPU])
        elif any(isinstance(a, OvMultipleDevice) and a.type == OvProviderType.NPU for a in available):
            return OvProviders([OvMultipleDevice(OvProviderType.NPU, 0), OvProviderType.CPU])
        elif any(isinstance(a, OvProviderType) and a == OvProviderType.GPU for a in available):
            return OvProviders([OvProviderType.GPU, OvProviderType.CPU])
        elif any(isinstance(a, OvProviderType) and a == OvProviderType.NPU for a in available):
            return OvProviders([OvProviderType.NPU, OvProviderType.CPU])
        elif any(isinstance(a, OvProviderType) and a == OvProviderType.CPU for a in available):
            return OvProviders([OvProviderType.CPU])
        else:
            return OvProviders([], flag=OvProvidersFlag.AUTO)

    @staticmethod
    def parse(value: str) -> "OvProviders":
        if not value or value == "default":
            return OvProviders.default()
        if value == "auto":
            return OvProviders([], OvProvidersFlag.AUTO)
        sp = value.split(':')
        flag = OvProvidersFlag.NONE
        if len(sp) >= 2:
            if sp[0] == "default":
                return OvProviders.default()
            elif sp[0] == "auto":
                flag = OvProvidersFlag.AUTO
            elif sp[0] == "multi":
                flag = OvProvidersFlag.MULTI
            elif sp[0] == "hetero":
                flag = OvProvidersFlag.HETERO
            else:
                raise ProviderParseError(f"未知的前缀: {sp[0]}")
            sp = sp[1:]
        devs: list[OvProviderType | OvMultipleDevice] = []
        sps = sp[0].split(' ')
        for a in sps:
            sp = a.split('.')
            if len(sp) == 1:
                if sp[0] == "gpu":
                    devs.append(OvProviderType.GPU)
                elif sp[0] == "npu":
                    devs.append(OvProviderType.NPU)
                elif sp[0] == "cpu":
                    devs.append(OvProviderType.CPU)
                else:
                    raise ProviderParseError(f"未知的设备类型: {sp[0]}")
            else:
                if sp[0] == "gpu":
                    devs.append(OvMultipleDevice(OvProviderType.GPU, int(sp[1])))
                elif sp[0] == "npu":
                    devs.append(OvMultipleDevice(OvProviderType.NPU, int(sp[1])))
                elif sp[0] == "cpu":
                    devs.append(OvMultipleDevice(OvProviderType.CPU, int(sp[1])))
                else:
                    raise ProviderParseError(f"未知的设备类型: {sp[0]}")
        res = OvProviders(devs, flag)
        return res