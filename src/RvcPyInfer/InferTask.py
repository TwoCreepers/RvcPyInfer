from typing import List, Tuple, Callable, TYPE_CHECKING
from pathlib import Path

import soundfile
import numpy as np
from numpy.typing import NDArray

from .type_alist import FileLike, Audio, AudioLike, F0ExtractAlgorithm, F0ExtractAlgorithmList
from .f0_utils import build_f0extract_func, apply_rise_tone, f0_to_mel, normalized_mel

if TYPE_CHECKING:
    from .RvcContext import RvcContext

class InferTask:
    audios: List[Audio]
    def __init__(self, 
                 context: "RvcContext",
                 vec_model: Path,
                 gen_model: Tuple[Path, int],
                 *audios: AudioLike,
                 f0extract_algorithm: F0ExtractAlgorithm,
                 f0_up_semitone: float = 0,
                 f0_min: float = 50,
                 f0_max: float = 1100,
                 ) -> None:
        self.context = context
        self.vec = vec_model
        self.gen = gen_model
        self.audio_likes = list(audios)
        self.f0extract_algorithm = f0extract_algorithm
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_up_semitone = f0_up_semitone

    def read(self) -> None:
        self.audios = []
        for audio_like in self.audio_likes:
            if isinstance(audio_like, Tuple) and len(audio_like) == 2 and isinstance(audio_like[0], np.ndarray) and isinstance(audio_like[1], int):
                self.audios.append(audio_like)
            else:
                assert not isinstance(audio_like, Tuple) # 这个类型检查真的很奇怪有没有
                data, sr = soundfile.read(
                    audio_like, dtype="float32"
                )
                data = data.astype(np.float32)
                if len(data.shape) != 1:
                    data = data.mean(axis=1, dtype=np.float32)
                self.audios.append((data, sr))
        del self.audio_likes

    def hubert_extract(self) -> None:
        with self.context._vec_pool.borrow(self.vec) as model:
            self.hubert_list: List[NDArray[np.float32]] = [model.infer(a) for a in self.audios]

    def f0extract(self) -> None:
        assert self.f0extract_algorithm in F0ExtractAlgorithmList
        func = build_f0extract_func(
            self.f0extract_algorithm, # pyright: ignore[reportArgumentType]
            f0_min=self.f0_min,
            f0_max=self.f0_max
        )
        self.f0_list = [func(a, len(h)) for a, h in zip(self.audios, self.hubert_list)]

    def apply_rise_tone(self) -> None:
        self.f0_list = [apply_rise_tone(f0, self.f0_up_semitone) for f0 in self.f0_list]

    def build_mel(self) -> None:
        self.mel_list = [f0_to_mel(f0) for f0 in self.f0_list]
        del self.audios

    def normalized_mel(self) -> None:
        min = f0_to_mel(self.f0_min)
        max = f0_to_mel(self.f0_max)
        self.mel_list = [normalized_mel(mel, min, max) for mel in self.mel_list]
        self.mel_i64_list = [np.rint(mel).astype(np.int64) for mel in self.mel_list]
        del self.mel_list

    def gen_infer(self, callback: Callable[[int, Audio], None], sid: int = 0, seed: int | None = 1234) -> None:
        with self.context._gen_pool.borrow(self.gen) as model:
            for i in range(len(self.hubert_list)):
                hubert = self.hubert_list[i]
                f0 = self.f0_list[i]
                mel = self.mel_i64_list[i]
                
                res = model.infer(
                    hubert,
                    mel,
                    f0,
                    sid,
                    seed
                )
                res_d, res_sr = res
                source_d, source_sr = self.audios[i]

                source_len = len(source_d)
                res_len = source_len * res_sr // source_sr
                
                pad_len = res_len - len(res_d)
                if pad_len == 0:
                    callback(i, res)
                elif pad_len < 0:
                    callback(i, (res_d[:res_len], res_sr))
                else: # 反正原项目就是这么做的
                    callback(i, (np.pad(
                        res_d,
                        [0, pad_len],
                        mode="constant",
                    ), res_sr))

    def pro_infer(self) -> None:
        self.hubert_extract()
        self.f0extract()
        self.apply_rise_tone()
        self.build_mel()
        self.normalized_mel()
                
    def run(self, sid: int = 0, seed: int | None = 1234) -> List[Audio]:
        self.read()
        self.pro_infer()

        res = []
        self.gen_infer(
            lambda _, a: res.append(a),
            sid, seed
        )
        return res
    
    def run_and_save(self, *output: FileLike, 
                     subtype: str | None = None,
                     format: str | None = None,
                     sid: int = 0, 
                     seed: int | None = 1234):
        self.read()

        if len(self.audios) != len(output):
            raise ValueError(f"音频数量 {len(self.audios)} 与输出数量 {len(output)} 不匹配")

        self.pro_infer()

        def save(i: int, a: Audio) -> None:
            d, s = a
            soundfile.write(output[i], 
                            d, s, 
                            subtype=subtype,
                            format=format)
        self.gen_infer(
            save, sid, seed
        )