import warnings
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

from ..warn.InferModelWarn import InferModelWarn
from ..error.InferEnvError import InferEnvError
from ..InferProviders import InferProviders

if TYPE_CHECKING:
    from ..infer_env import HAS_ORT, HAS_OPENVINO
    if HAS_ORT:
        import onnxruntime as ort # pyright: ignore[reportMissingImports]
    if HAS_OPENVINO:
        import openvino # pyright: ignore[reportMissingImports]

def load_model(path: Path, providers: InferProviders) -> "Tuple[ort.InferenceSession | None, openvino.CompiledModel | None]": # pyright: ignore[reportAttributeAccessIssue]
    session = None
    compiled_model = None
    if providers.is_ort():
            import onnxruntime as ort # pyright: ignore[reportMissingImports]
            session = ort.InferenceSession(
                path.with_suffix(".onnx"), providers=providers.get_onnx_provider()
            )
    elif providers.is_ov():
        from openvino import save_model, convert_model # pyright: ignore[reportMissingImports]
        from ..ov.OVCoreSingleton import core
        if (path.parent / path.stem).with_suffix(".xml").exists():
            model = core.read_model(str((path.parent / path.stem).with_suffix(".xml").resolve()))
            compiled_model = core.compile_model(model=model, device_name=providers.get_openvino_device())
        elif (path.parent / path.stem).with_suffix(".onnx").exists():
            warnings.warn("您正在尝试使用 OpenVINO 运行 onnx，模型将自动转换并存档", InferModelWarn)
            model = convert_model(str((path.parent / path.stem).with_suffix(".onnx").resolve()))
            save_model(model, str((path.parent / path.stem).with_suffix(".xml").resolve()), compress_to_fp16=False)
            compiled_model = core.compile_model(model=model, device_name=providers.get_openvino_device())
        else:
            raise FileNotFoundError(f"找不到模型: {path.parent / path.stem}")
    else:
        raise InferEnvError("没有任意一个支持的推理引擎或提供者 Flag 错误，请至少安装一个可选模块")
    
    return session, compiled_model