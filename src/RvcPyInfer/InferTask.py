from typing import List, Tuple, Callable, TYPE_CHECKING
from pathlib import Path

import soundfile
import numpy as np
from numpy.typing import NDArray

from .type_alist import FileLike, Audio, AudioLike, F0ExtractAlgorithm, F0ExtractAlgorithmList
from .f0_utils import build_f0extract_func, apply_rise_tone, f0_to_mel, normalized_mel
from .audio.audio_utils import split_by_silence, split_by_max_len_with_overlap, crossfade

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
                 slice_max_len = 30,
                 slice_overlap_len = 5,
                 ) -> None:
        self.context = context
        self.vec = vec_model
        self.gen = gen_model
        self.audio_likes = list(audios)
        self.f0extract_algorithm = f0extract_algorithm
        self.f0_up_semitone = f0_up_semitone
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.slice_max_len = slice_max_len
        self.slice_overlap_len = slice_overlap_len

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

    def chunk_infer(self, audio: Audio, f0extract_func: Callable[[Audio, int], NDArray[np.float32]], sid: int = 0, seed: int | None = 1234) -> Audio:
        model = self.context._vec_pool.get(self.vec)
        hubert = model.infer(audio)

        f0 = f0extract_func(audio, hubert.shape[0])
        f0 = apply_rise_tone(f0, self.f0_up_semitone)

        mel = normalized_mel(f0_to_mel(f0), f0_to_mel(self.f0_min), f0_to_mel(self.f0_max))
        mel = np.rint(mel).astype(np.int64)

        model = self.context._gen_pool.get(self.gen)
        res = model.infer(
            phone=hubert,
            f0_pitch=f0, mel_pitch=mel,
            sid=sid, seed=seed
        )

        res_d, res_sr = res
        source_d, source_sr = audio

        source_len = len(source_d)
        res_len = source_len * res_sr // source_sr
        
        pad_len = res_len - len(res_d)
        if pad_len == 0:
            return res
        elif pad_len < 0:
            return (res_d[:res_len], res_sr)
        else:
            return (
                np.pad(res_d, [0, pad_len], mode="constant"),
                res_sr
            )

    def gen_infer(self, callback: Callable[[int, Audio], None], sid: int = 0, seed: int | None = 1234) -> None:
        func = build_f0extract_func(self.f0extract_algorithm, self.f0_min, self.f0_max) # pyright: ignore[reportArgumentType]
        for i, audio in enumerate(self.audios):
            splited = split_by_max_len_with_overlap(
                audio,
                max_len=self.slice_max_len,
                overlap_len=self.slice_overlap_len
                )
            del audio # 节约点内存
            def handle(c):
                return self.chunk_infer(c, func, sid, seed)
            infer = [handle(i) for i in splited]
            infer = list(infer)
            res = crossfade(
                infer,
                overlap_len=self.slice_overlap_len
                )
            callback(i, res)
                
    def run(self, sid: int = 0, seed: int | None = 1234) -> List[Audio]:
        self.read()

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
        def save(i: int, a: Audio) -> None:
            d, s = a
            soundfile.write(output[i], 
                            d, s, 
                            subtype=subtype,
                            format=format)
        self.gen_infer(
            save, sid, seed
        )