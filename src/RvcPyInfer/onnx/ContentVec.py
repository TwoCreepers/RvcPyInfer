import numpy as np
from numpy.typing import NDArray

from ..audio.audio_utils import reSR
from ..error.InferEnvError import InferEnvError
from ..InferProviders import InferProviders
from ..path_utils import path as pathf
from ..type_alist import Audio, PathLike
from .model_loader import load_model


class ContentVec:
    def __init__(self, path: PathLike, providers: InferProviders) -> None:
        self.session, self.compiled_model = load_model(pathf(path), providers)

    def infer(self, audio: Audio) -> NDArray[np.float32]:
        data, sr = audio
        assert len(data.shape) == 1, "输入音频应为单通道"
        data, sr = reSR(audio, target_sr=16000)
        data = data.reshape(1, 1, -1)
        input = {
            "source": data
        }
        logits = None
        if self.session is not None:
            logits = self.session.run(
                    output_names=["embed"], 
                    input_feed=input)[0]
            assert isinstance(logits, np.ndarray), "ort 输出应为 NDArray"
        elif self.compiled_model is not None:
            infer_request = self.compiled_model.create_infer_request()
            result_dict = infer_request.infer(inputs=input)
            logits = result_dict["embed"]
            del infer_request, result_dict
        else:
            raise InferEnvError("没有任意一个支持的推理引擎，请至少安装一个可选模块")
        # (1, frames, embed_size)
        return np.repeat(logits, 2, axis=1).squeeze(axis=0).astype(np.float32) # 因为模型帧长为 320, 这里复制一份到 160