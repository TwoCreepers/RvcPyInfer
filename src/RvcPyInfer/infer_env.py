import warnings

from .warn.InferEnvWarn import InferEnvWarn

try:
    import onnxruntime # pyright: ignore[reportMissingImports]
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import openvino # pyright: ignore[reportMissingImports]
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

HAS_ONE = HAS_ORT or HAS_OPENVINO

if not HAS_ONE:
    warnings.warn("没有任意一个支持的推理引擎，请至少安装一个可选模块", InferEnvWarn)