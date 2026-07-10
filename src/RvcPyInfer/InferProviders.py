from typing import List
from enum import Flag, auto

from .error.InferEnvError import InferEnvError

class InferProviders(Flag):
    NONE = auto() # 仅用于内部方便处理

    # ort
    ORT_CPU = auto()
    FORCE_CUDA = auto()
    FORCE_DML = auto()
    CUDA = ORT_CPU | FORCE_CUDA
    DML = ORT_CPU | FORCE_DML

    # openvino
    OPENVINO_AUTO = auto()
    OPENVINO_CPU = auto()
    OPENVINO_GPU = auto()
    OPENVINO_NPU = auto()
    OPENVINO_OPTIONAL_MULTI  = auto()
    OPENVINO_OPTIONAL_HETERO = auto()
    OPENVINO_ALL = OPENVINO_CPU | OPENVINO_GPU | OPENVINO_NPU | OPENVINO_OPTIONAL_MULTI

    def is_ort(self) -> bool:
        return bool((InferProviders.ORT_CPU | InferProviders.CUDA | InferProviders.DML) & self)
    def is_ov(self) -> bool:
        return bool((InferProviders.OPENVINO_CPU | InferProviders.OPENVINO_GPU | InferProviders.OPENVINO_NPU | InferProviders.OPENVINO_AUTO) & self)

    def get_onnx_provider(self) -> List[str]:
        res = []
        if self.FORCE_CUDA in self:
            res.append("CUDAExecutionProvider")
        if self.FORCE_DML in self:
            res.append("DmlExecutionProvider")
        if self.ORT_CPU in self:
            res.append("CPUExecutionProvider")
        return res
    
    def get_openvino_device(self) -> str:
        devices = []
        if self.OPENVINO_GPU in self:
            # 现场找全
            from openvino import Core # pyright: ignore[reportMissingImports]
            devices.extend(filter(lambda name: name.startswith("GPU"), Core().available_devices))
        if self.OPENVINO_NPU in self:
            devices.append("NPU")
        if self.OPENVINO_CPU in self:
            devices.append("CPU")
        devices_str = ",".join(devices)
        if self.OPENVINO_AUTO in self:
            if len(devices) != 0:
                return f"AUTO:{devices_str}"
            else:
                return "AUTO"
        elif self.OPENVINO_OPTIONAL_MULTI in self:
            return f"MULTI:{devices_str}"
        elif self.OPENVINO_OPTIONAL_HETERO in self:
            return f"HETERO:{devices_str}"
        else:
            return devices[0]

    @staticmethod
    def default() -> "InferProviders":
        available = InferProviders.currently_available()

        if InferProviders.FORCE_CUDA in available:
            return InferProviders.CUDA
        elif InferProviders.FORCE_DML in available:
            return InferProviders.DML
        elif InferProviders.OPENVINO_CPU in available:
            # 有时候自动的异构并行反而比纯 CPU 还慢，这里保守点
            return InferProviders.OPENVINO_CPU
        elif InferProviders.ORT_CPU in available:
            return InferProviders.ORT_CPU
        else:
            raise InferEnvError("没有任意一个支持的推理引擎，请至少安装一个可选模块")
        
    @staticmethod
    def currently_available() -> "InferProviders":
        from .infer_env import HAS_ONE, HAS_ORT, HAS_OPENVINO
        res = InferProviders.NONE

        if HAS_ORT:
            import onnxruntime as ort # pyright: ignore[reportMissingImports]
            available_providers = ort.get_available_providers()
            if "CPUExecutionProvider" in available_providers:
                res = res | InferProviders.ORT_CPU
            if "CUDAExecutionProvider" in available_providers:
                res = res | InferProviders.FORCE_CUDA
            if "DmlExecutionProvider" in available_providers:
                res = res | InferProviders.FORCE_DML
        if HAS_OPENVINO:
            from openvino import Core # pyright: ignore[reportMissingImports]
            devices = Core().available_devices
            res = res | InferProviders.OPENVINO_AUTO | InferProviders.OPENVINO_OPTIONAL_MULTI | InferProviders.OPENVINO_OPTIONAL_HETERO
            if "CPU" in devices:
                res = res | InferProviders.OPENVINO_CPU
            if any(i.startswith("GPU") for i in devices):
                res = res | InferProviders.OPENVINO_GPU
            if "NPU" in devices:
                res = res | InferProviders.OPENVINO_NPU
        if not HAS_ONE: 
            raise InferEnvError("没有任意一个支持的推理引擎，请至少安装一个可选模块")
        
        assert res != InferProviders.NONE, "我算是知道了，你们各个都身怀绝技"
        res = res & ~InferProviders.NONE
        return res