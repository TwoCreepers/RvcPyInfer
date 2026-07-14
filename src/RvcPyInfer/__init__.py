from . import f0_utils
from ._version import __version__  # noqa: F401
from .audio import audio_utils
from .cli import main as ModelInferCLI
from .error.InferEnvError import InferEnvError
from .error.NotSupportedAlgorithmError import NotSupportedAlgorithmError
from .InferProviders import InferProviders
from .InferTask import InferTask
from .onnx.ContentVec import ContentVec
from .onnx.model import Exporter as OnnxExporter
from .onnx.model.direct_read_sr import direct_read_sr
from .onnx.model.cli import main as ModelExportToolCLI
from .onnx.ModelSimplePool import ModelSimplePool
from .onnx.RvcGen import RvcGen
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
    "RvcContext",
    "InferTask",
    "OnnxExporter",
    "direct_read_sr",
    "ModelExportToolCLI",
    "ContentVec",
    "RvcGen",
    "ModelSimplePool",
    "ModelInferCLI"
]