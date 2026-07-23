from . import f0_utils
from ._version import __version__  # noqa: F401
from .audio import audio_utils
from .cli import main as ModelInferCLI
from .error.InferEnvError import InferEnvError
from .error.NotSupportedAlgorithmError import NotSupportedAlgorithmError
from .InferTask import InferTask
from .onnx.ContentVec import ContentVec
from .onnx.model.cli import main as ModelExportToolCLI
from .onnx.model.direct_read_sr import direct_read_sr
from .onnx.model.Exporter import Exporter as OnnxExporter
from .onnx.model.Optimizer import Optimizer as OnnxOptimizer
from .onnx.ModelSimplePool import ModelSimplePool
from .onnx.RvcGen import RvcGen
from .provider.InferProviders import InferProviders
from .provider.OrtProviders import OrtProviders
from .provider.OvProviders import OvProviders
from .provider.provider_type_alist import ProvidersLike
from .provider.provider_utils import infer_providers
from .RvcContext import RvcContext
from .type_alist import Audio, AudioLike, F0ExtractAlgorithm, F0ExtractAlgorithmList, FileLike, PathLike
from .warn.InferEnvWarn import InferEnvWarn
from .warn.InferModelWarn import InferModelWarn
from .warn.InferWarn import InferWarn

# 用户肯定不会希望我的 __version__ 去污染 ta 的版本号

__all__ = [
    "Audio", "AudioLike", "PathLike", "FileLike", "F0ExtractAlgorithm", "F0ExtractAlgorithmList",
    "InferEnvError", 
    "NotSupportedAlgorithmError",
    "InferWarn",
    "InferEnvWarn",
    "InferModelWarn",
    "f0_utils",
    "audio_utils",
    "InferProviders",
    "OrtProviders",
    "OvProviders",
    "ProvidersLike",
    "infer_providers",
    "RvcContext",
    "InferTask",
    "OnnxExporter",
    "OnnxOptimizer",
    "direct_read_sr",
    "ModelExportToolCLI",
    "ContentVec",
    "RvcGen",
    "ModelSimplePool",
    "ModelInferCLI"
]