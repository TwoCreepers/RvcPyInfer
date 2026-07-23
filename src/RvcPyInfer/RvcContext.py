import warnings
from pathlib import Path

from .index.RvcFeatIndex import RvcFeatIndex
from .InferTask import InferTask
from .onnx.ContentVec import ContentVec
from .onnx.f0.RmvpeModel import RmvpeModel
from .onnx.ModelSimplePool import ModelSimplePool
from .onnx.RvcGen import RvcGen
from .path_utils import path
from .provider.InferProviders import InferProviders
from .provider.provider_type_alist import ProvidersLike
from .provider.provider_utils import infer_providers
from .type_alist import AudioLike, F0ExtractAlgorithm, PathLike
from .warn.InferEnvWarn import InferEnvWarn


# 请注意，RvcContext 并不是线程安全的
class RvcContext:
    _rmvpe: RmvpeModel | None
    def __init__(self,
                 providers: ProvidersLike = InferProviders.default(),
                 rmvpe: PathLike | None = None,
                 vec_pool_permanent_size: int = 1,
                 gen_pool_permanent_size: int = 2,
                 index_pool_permanent_size: int = 2) -> None:
        self._providers = infer_providers(providers)

        self._rmvpe_path = rmvpe
        self._rmvpe = None

        self._vec_pool = ModelSimplePool[Path, ContentVec](
            lambda p: ContentVec(p.resolve(), self._providers),
            vec_pool_permanent_size
        )        
        self._gen_pool = ModelSimplePool[tuple[Path, int], RvcGen](
            lambda args: RvcGen(args[0].resolve(), args[1], self._providers),
            gen_pool_permanent_size,
        )

        from .infer_env import HAS_FAISS
        if HAS_FAISS:
            self._index_pool = ModelSimplePool[Path, RvcFeatIndex](
                lambda p: RvcFeatIndex(p.resolve()),
                index_pool_permanent_size
            )
        else:
            self._index_pool = None

    def _get_rmvpe(self) -> RmvpeModel:
        if self._rmvpe_path is None:
            raise ValueError("未配置 rmvpe 模型路径，但尝试使用 rmvpe")
        if self._rmvpe is None:
            self._rmvpe = RmvpeModel(self._rmvpe_path, self._providers)
        return self._rmvpe

    def clear(self) -> None:
        self._rmvpe = None

        self._vec_pool.clear()
        self._gen_pool.clear()

    def build_task(self, 
            # -- 必填 --
            vec_model: PathLike, 
            gen_model: PathLike, 
            gen_model_sr: int,
            *audios: AudioLike,

            # 可选扩展
            sid: int = 0,
            seed: int | None = 1234,

            # -- 特征索引 --
            index_path: PathLike | None = None,
            index_rate: float = 0.33,
            index_k: int = 8, # 检索的最近特征数量
            index_consonant_protect: float = 0.66, # 对辅音的特征进行保护，减少掺入的特征

            # -- f0 提取 --
            f0extract_algorithm: F0ExtractAlgorithm = "dio",
            f0_up_semitone: float = 0,
            f0_min: float = 50,
            f0_max: float = 1100,
            f0_median_filter_win_size: int = -1, # 小于 3 静默关闭

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
            rms_gain_clip: float = 5.0,
            ) -> "InferTask":
        vec = path(vec_model)
        gen = path(gen_model)

        if not vec.exists():
            raise FileNotFoundError(f"找不到模型: {vec}")
        if not gen.exists():
            raise FileNotFoundError(f"找不到模型: {gen}")
        
        if index_path is not None:
            from .infer_env import HAS_FAISS
            if not HAS_FAISS:
                warnings.warn("您正在使用特征索引，但并未安装 faiss。如果您想使用特征索引请安装 [index] 可选依赖", InferEnvWarn)
                index_path = None
            else:
                index_path = path(index_path)
                if not index_path.exists():
                    raise FileNotFoundError(f"找不到特征索引: {index_path}")
                
        if f0_median_filter_win_size >= 3: # 启用条件
            if f0_median_filter_win_size % 2 == 0: 
                raise ValueError(f"基频中值滤波窗口大小必须是奇数: {f0_median_filter_win_size}")

        return InferTask(
            self,
            vec,
            (gen, gen_model_sr),
            *audios,
            sid=sid,
            seed=seed,
            index_path=index_path,
            index_rate=index_rate,
            index_k=index_k,
            index_consonant_protect=index_consonant_protect,
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
            rms_gain_clip=rms_gain_clip,
        )