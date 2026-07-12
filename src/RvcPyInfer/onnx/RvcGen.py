import numpy as np
from numpy.typing import NDArray

from ..error.InferEnvError import InferEnvError
from ..InferProviders import InferProviders
from ..path_utils import path as pathf
from ..type_alist import Audio, PathLike
from .model_loader import load_model


class RvcGen:
    def __init__(self, path: PathLike, model_out_sr: int, providers: InferProviders) -> None:
        self.sr = model_out_sr
        self.session, self.compiled_model = load_model(pathf(path), providers)
        if self.session is not None:
            self.channels = next(filter(lambda arg: arg.name == "rnd", self.session.get_inputs())).shape[1]
        elif self.compiled_model is not None:
            self.channels = next(filter(lambda arg: arg.any_name == "rnd", self.compiled_model.inputs)).partial_shape[1].get_length()

    def infer(self, phone: NDArray[np.float32], mel_pitch: NDArray[np.int64], f0_pitch: NDArray[np.float32], sid: int = 0, seed: int | None = 1234) -> Audio:
        assert phone.shape[0] == mel_pitch.shape[0] == f0_pitch.shape[0], "帧数应统一"
        phone_len = phone.shape[0]
        ds = np.array([sid], dtype=np.int64)
        rnd = np.random.default_rng(seed).standard_normal((1, self.channels, phone_len), dtype=np.float32)
        phone = phone[None, ...]
        mel_pitch = mel_pitch.reshape(1, -1)
        f0_pitch = f0_pitch.reshape(1, -1)

        model_input = {
            "rnd": rnd,
            "ds": ds,
            "phone": phone,
            "phone_lengths": np.array([phone_len]),
            "pitch": mel_pitch,
            "pitchf": f0_pitch
        }
        if self.session is not None:
            output = self.session.run(
                output_names=["audio"],
                input_feed=model_input
            )[0]
        elif self.compiled_model is not None:
            infer_request = self.compiled_model.create_infer_request()
            result_dict = infer_request.infer(inputs=model_input)
            output = result_dict["audio"]
            del infer_request, result_dict
        else:
            raise InferEnvError("没有任意一个支持的推理引擎，请至少安装一个可选模块")
        # (1, sampling)
        assert isinstance(output, np.ndarray), "输出应为 NDArray"
        return output.squeeze(axis=1).squeeze(axis=0).astype(np.float32), self.sr