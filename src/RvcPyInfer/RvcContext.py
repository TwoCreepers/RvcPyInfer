from pathlib import Path
from typing import Tuple

from .InferProviders import InferProviders
from .onnx.ModelSimplePool import ModelSimplePool
from .onnx.ContentVec import ContentVec
from .onnx.RvcGen import RvcGen
from .type_alist import AudioLike, PathLike, F0ExtractAlgorithm
from .path_utils import path
from .InferTask import InferTask

# 请注意，RvcContext 并不是线程安全的
class RvcContext:
    def __init__(self,
                 providers: InferProviders = InferProviders.default(),
                 vec_pool_permanent_size: int = 1,
                 gen_pool_permanent_size: int = 2,) -> None:
        self._providers = providers
        self._vec_pool = ModelSimplePool[Path, ContentVec](
            lambda p: ContentVec(p.resolve(), self._providers),
            vec_pool_permanent_size
        )        
        self._gen_pool = ModelSimplePool[Tuple[Path, int], RvcGen](
            lambda args: RvcGen(args[0].resolve(), args[1], self._providers),
            gen_pool_permanent_size,
        )

    def clear_pool(self):
        self._vec_pool.clear()
        self._gen_pool.clear()

    def build_task(self, 
                # -- 必填 --
                vec_model: PathLike, 
                gen_model: PathLike, 
                gen_model_sr: int,
                *audios: AudioLike,

                # -- f0 提取 --
                f0extract_algorithm: F0ExtractAlgorithm = "dio",
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
                ) -> "InferTask":
        vec = path(vec_model)
        gen = path(gen_model)

        if not vec.exists():
            raise FileNotFoundError(f"找不到模型: {vec}")
        if not gen.exists():
            raise FileNotFoundError(f"找不到模型: {gen}")

        return InferTask(
            self,
            vec,
            (gen, gen_model_sr),
            *audios,
            f0extract_algorithm=f0extract_algorithm,
            f0_up_semitone=f0_up_semitone,
            f0_min=f0_min,
            f0_max=f0_max,
            slice_max_len=slice_max_len,
            slice_overlap_len=slice_overlap_len,
            silence_frame_len=silence_frame_len,
            silence_hop_len=silence_hop_len,
            silence_thresh_db=silence_thresh_db,
            silence_min_silence_duration_ms=silence_min_silence_duration_ms,
            silence_max_transition_ms=silence_max_transition_ms,
            rms_match_frame_len=rms_match_frame_len,
            rms_match_hop_len=rms_match_hop_len,
            rms_match_mix=rms_match_mix,
        )