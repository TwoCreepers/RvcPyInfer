from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile
from numpy.typing import NDArray

from .audio.audio_utils import crossfade, reSR, rms_frame_match, split_by_max_len_with_overlap, split_by_silence
from .f0_utils import apply_rise_tone, build_f0extract_func, f0_to_mel, normalized_mel
from .type_alist import Audio, AudioLike, F0ExtractAlgorithm, F0ExtractAlgorithmList, FileLike

if TYPE_CHECKING:
    from .RvcContext import RvcContext

class InferTask:
    audios: list[Audio]
    def __init__(self, 
            # -- 必填 --
            context: "RvcContext",
            vec_model: Path,
            gen_model: tuple[Path, int],
            *audios: AudioLike,

            # 可选扩展
            sid: int = 0,
            seed: int | None = 1234,

            # -- 特征索引 --
            index_path: Path | None = None,
            index_rate: float = 0.33,
            index_k: int = 8, # 检索的最近特征数量

            # -- f0 提取 --
            f0extract_algorithm: F0ExtractAlgorithm,
            f0_up_semitone: float = 0,
            f0_min: float = 50,
            f0_max: float = 1100,

            # -- 强制切片与交叉淡化 --
            slice_max_len: int = 30,
            slice_overlap_len: int = 5,

            # -- 静音切片 --
            silence_frame_len: int = 20,
            silence_hop_len: int = 10,
            silence_thresh_db: float = -40,
            silence_min_silence_duration_ms: float = 800.0,
            silence_max_transition_ms: float = 100.0,

            # -- RMS 包络匹配 --
            rms_match_frame_len: int = 20,
            rms_match_hop_len: int = 10,
            rms_match_mix: float = 1.0, # 一般不用改
            rms_gain_clip: float = 5.0, # 用来解决原音频底噪造成的伪 rms 包络问题
            ) -> None:
        self.context = context
        self.vec = vec_model.resolve()
        self.gen = (gen_model[0].resolve(), gen_model[1])
        self.audio_likes = list(audios)
        self.sid = sid
        self.seed = seed
        self.index_path = index_path
        self.index_rate = index_rate
        self.index_k = index_k
        self.f0extract_algorithm = f0extract_algorithm
        self.f0_up_semitone = f0_up_semitone
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.slice_max_len = slice_max_len
        self.slice_overlap_len = slice_overlap_len
        self.silence_frame_len = silence_frame_len
        self.silence_hop_len = silence_hop_len
        self.silence_thresh_db = silence_thresh_db
        self.silence_min_silence_duration_ms = silence_min_silence_duration_ms
        self.silence_max_transition_ms = silence_max_transition_ms
        self.rms_match_frame_len = rms_match_frame_len
        self.rms_match_hop_len = rms_match_hop_len
        self.rms_match_mix = rms_match_mix
        self.rms_gain_clip = rms_gain_clip

    def read(self) -> None:
        self.audios = []
        for audio_like in self.audio_likes:
            if isinstance(audio_like, tuple) and len(audio_like) == 2 and isinstance(audio_like[0], np.ndarray) and isinstance(audio_like[1], int):
                self.audios.append(audio_like)
            else:
                assert not isinstance(audio_like, tuple) # 这个类型检查真的很奇怪有没有
                data, sr = soundfile.read(
                    audio_like, dtype="float32"
                )
                data = data.astype(np.float32)
                if len(data.shape) != 1:
                    data = data.mean(axis=1, dtype=np.float32)
                self.audios.append((data, sr))
        del self.audio_likes

    def core_chunk_infer(self, audio: Audio, f0extract_func: Callable[[Audio, int], NDArray[np.float32]]) -> Audio:
        model = self.context._vec_pool.get(self.vec)
        feats = model.infer(audio)

        f0 = f0extract_func(audio, feats.shape[0]) # 这里是没有批处理维度的
        f0 = apply_rise_tone(f0, self.f0_up_semitone)

        mel = normalized_mel(f0_to_mel(f0), f0_to_mel(self.f0_min), f0_to_mel(self.f0_max))
        mel = np.rint(mel).astype(np.int64)

        if self.index_rate > 1e-6 and self.index_path is not None and self.context._index_pool is not None:
            index = self.context._index_pool.get(self.index_path)
            feats = index.apply_index(feats, self.index_rate, self.index_k)

        model = self.context._gen_pool.get(self.gen)
        res = model.infer(
            phone=feats,
            f0_pitch=f0, mel_pitch=mel,
            sid=self.sid, seed=self.seed
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
    
    def chunk_infer(self, audio: Audio, f0extract_func: Callable[[Audio, int], NDArray[np.float32]]) -> Audio:
        splited = split_by_max_len_with_overlap( # 就算不足一个切片长也不会报错哒
            audio,
            max_len=self.slice_max_len,
            overlap_len=self.slice_overlap_len
            )
        del audio # 节约点内存
        def handle(c):
            return self.core_chunk_infer(c, f0extract_func)
        infer = [handle(i) for i in splited]
        res = crossfade(
            infer,
            overlap_len=self.slice_overlap_len
            )
        return res

    def gen_infer(self, callback: Callable[[int, Audio], None]) -> None:
        assert self.f0extract_algorithm in F0ExtractAlgorithmList, f"{self.f0extract_algorithm} 不是受支持的 f0 提取算法"
        func = build_f0extract_func(self.f0extract_algorithm, self.f0_min, self.f0_max) # pyright: ignore[reportArgumentType]
        for i, audio in enumerate(self.audios):
            splited = split_by_silence(
                audio,
                frame_len=self.silence_frame_len,
                hop_len=self.silence_hop_len,
                silence_thresh_db=self.silence_thresh_db,
                min_silence_duration_ms=self.silence_min_silence_duration_ms,
                max_transition_ms=self.silence_max_transition_ms
            )
            chunks: list[Audio] = []
            for chunk, is_sil in splited:
                if is_sil:
                    chunks.append(reSR(chunk, target_sr=self.gen[1])) # 内部判断相等会直接返回
                else:
                    chunks.append(self.chunk_infer(chunk, func))
            datas = [data for data, sr in chunks]
            res = (np.concatenate(datas), self.gen[1])
            res = rms_frame_match(
                source=res,
                target=reSR(audio, target_sr=res[1]),
                frame_len=self.rms_match_frame_len,
                hop_len=self.rms_match_hop_len,
                mix=self.rms_match_mix,
                gain_clip=self.rms_gain_clip
            )
            callback(i, res)
                
    def run(self) -> list[Audio]:
        self.read()

        res = []
        self.gen_infer(
            lambda _, a: res.append(a)
        )
        return res
    
    def run_and_save(self, *output: FileLike, 
                     subtype: str | None = None,
                     format: str | None = None):
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
            save
        )

    def run_and_callback(self, 
                        callback: Callable[[int, Audio], None]) -> None:
        self.read()
        self.gen_infer(
            callback=callback
        )