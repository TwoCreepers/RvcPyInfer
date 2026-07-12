import math
from collections.abc import Callable
from typing import overload

import numpy as np
import pyworld as pw
from numpy.typing import NDArray

from .error.NotSupportedAlgorithmError import NotSupportedAlgorithmError
from .type_alist import Audio, F0ExtractAlgorithm


def interpolate_f0[T: np.floating](f0: NDArray[T]) -> NDArray[T]:
    """
    对F0进行插值处理
    """
    nonzero_idx = np.where(f0 > 0.0)[0]
    if len(nonzero_idx) == 0:
            return f0
    all_idx = np.arange(len(f0))
    ip_data = np.interp(all_idx, nonzero_idx, f0[nonzero_idx])
    ip_data = ip_data.astype(f0.dtype)
    return ip_data

def dio(audio: Audio, 
        p_len: int, 
        f0_min: float = 50.0, 
        f0_max: float = 1100.0,
        allowed_range: float = 0.1,
        channels_in_octave: float = 2.0) -> NDArray[np.float32]:
    data, sr = audio
    data = data.astype(np.float64)
    frame_period = len(data) * 1000 / sr / p_len
    f0, t = pw.dio( # pyright: ignore[reportAttributeAccessIssue]
        data,
        sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=frame_period,
        allowed_range=allowed_range,
        channels_in_octave=channels_in_octave
    ) 
    f0 = pw.stonemask( # pyright: ignore[reportAttributeAccessIssue]
        data,
        f0, t,
        sr
    )
    assert isinstance(f0, np.ndarray) and isinstance(t, np.ndarray), "pyworld 输出类型应是 NDArray" # 给类型注解看的
    f0, t = f0.astype(np.float32), t.astype(np.float32)
    f0 = np.round(f0, 1) # 不知道，但原项目就是这么做了，我只是一个做 onnx 推理的我怎么知道为什么要这样做
    f0 = interpolate_f0(f0)
    # 一般来说不太可能会发生长度不匹配的情况，但防御性编程
    pad_len = p_len - len(f0)
    if pad_len == 0:
        return f0
    elif pad_len > 0:
        return np.pad(f0, [0, pad_len], mode="constant", constant_values=f0[-1])
    else:
        return f0[:p_len]
         

def harvest(audio: Audio, 
        p_len: int, 
        f0_min: float = 50.0, 
        f0_max: float = 1100.0) -> NDArray[np.float32]:
    data, sr = audio
    data = data.astype(np.float64)
    frame_period = len(data) * 1000 / sr / p_len
    f0, t = pw.harvest( # pyright: ignore[reportAttributeAccessIssue]
        data,
        sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=frame_period
    ) 
    f0 = pw.stonemask( # pyright: ignore[reportAttributeAccessIssue]
        data,
        f0, t,
        sr
    )
    assert isinstance(f0, np.ndarray) and isinstance(t, np.ndarray), "pyworld 输出类型应是 NDArray" # 给类型注解看的
    f0, t = f0.astype(np.float32), t.astype(np.float32)
    f0 = interpolate_f0(f0)
    # 一般来说不太可能会发生长度不匹配的情况，但防御性编程
    pad_len = p_len - len(f0)
    if pad_len == 0:
        return f0
    elif pad_len > 0:
        return np.pad(f0, [0, pad_len], mode="constant", constant_values=f0[-1])
    else:
        return f0[:p_len]
    
def build_f0extract_func(algorithm: F0ExtractAlgorithm,
        f0_min: float = 50.0, 
        f0_max: float = 1100.0,
        allowed_range: float = 0.1,
        channels_in_octave: float = 2.0) -> Callable[[Audio, int], NDArray[np.float32]]:
    if algorithm == "dio":
        return lambda a, p_l: dio(a, p_l, f0_min=f0_min, f0_max=f0_max, allowed_range=allowed_range, channels_in_octave=channels_in_octave)
    elif algorithm == "harvest":
        return lambda a, p_l: harvest(a, p_l, f0_min=f0_min, f0_max=f0_max)
    else:
        raise NotSupportedAlgorithmError(f"不支持的 f0 提取算法: {algorithm}")
     
def apply_rise_tone[T: np.floating](f0: NDArray[T], up_semitone: float) -> NDArray[T]:
    return f0 * 2.0 ** (up_semitone / 12.0)

@overload
def f0_to_mel(f0: float) -> float:
    ...
@overload
def f0_to_mel[T: np.floating](f0: NDArray[T]) -> NDArray[T]:
    ...
def f0_to_mel[T: np.floating](f0: float | NDArray[T]) -> float | NDArray[T]:
    if isinstance(f0, float):
        return 1127.0 * math.log(1 + f0 / 700.0)
    else:
        return 1127 * np.log(1 + f0 / 700)
    
def normalized_mel[T: np.floating](mel: NDArray[T], mel_min: float = f0_to_mel(50.0), mel_max: float = f0_to_mel(1100.0)) -> NDArray[T]:
    mel[mel > 0.0] = (mel[mel > 0.0] - mel_min) * 254.0 / (
        mel_max - mel_min
    ) + 1.0
    mel = np.clip(mel, 1, 255)
    return mel