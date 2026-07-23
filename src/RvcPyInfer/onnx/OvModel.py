import warnings
from collections import OrderedDict
from pathlib import Path

from numpy.typing import NDArray

from ..provider.OvProviders import OvProviders
from ..warn.InferModelWarn import InferModelWarn
from .InferModel import InferModel


class OvModel(InferModel):
    def __init__(self, model: Path, providers: OvProviders) -> None:
        model = model.resolve()
        ir_path = model.with_suffix(".xml").resolve()
        from ..ov.OVCoreSingleton import core
        if ir_path.exists():
            ovmodel = core.read_model(str(ir_path))
        else:
            warnings.warn("您正在尝试使用 OpenVINO 运行 onnx，模型将自动转换并存档", InferModelWarn)
            from openvino import convert_model, save_model  # pyright: ignore[reportMissingImports]
            ovmodel = convert_model(str(model))
            save_model(ovmodel, str(ir_path), compress_to_fp16=False)
        self.model = core.compile_model(model=model, device_name=providers.to_providers())

    def infer(self, output_names: list[str], inputs: dict[str, NDArray]) -> OrderedDict[str, NDArray]:
        infer_request = self.model.create_infer_request()
        infed = infer_request.infer(inputs=inputs)
        result = OrderedDict[str, NDArray]()
        for n in output_names:
            result[n] = infed[n]
        return result