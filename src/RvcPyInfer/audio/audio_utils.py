from typing import Literal, List, Tuple

import samplerate

import numpy as np
from numpy.typing import NDArray

from ..type_alist import Audio

def reSR(orig: Audio, 
         target_sr: int, 
         algorithm: Literal["sinc_best", "sinc_medium", "sinc_fastest", "linear", "zero_order_hold"] = "sinc_medium") -> Audio:
    orig_data, orig_sr = orig
    ratio = target_sr / orig_sr
    if target_sr == orig_sr:
        target_data = orig_data
    else:
        target_data: NDArray = samplerate.resample(orig_data, ratio, algorithm)
        assert isinstance(target_data, np.ndarray), "这库不对吧？返回值不是 ndarray"
    target_data = target_data.astype(np.float32)
    return target_data, target_sr

def frame_rms(audio: Audio, frame_len = 20, hop_len = 10) -> NDArray[np.float32]:
    """
    分帧计算RMS能量
    
    参数:
        audio: 输入音频
        frame_len: 帧长度(ms)
        hop_len: 帧移(ms)
        
    返回:
        rms_values: 每帧的RMS值
    """
    data, sr = audio

    frame_size = int(round(frame_len * sr / 1000))
    hop_size = int(round(hop_len * sr / 1000))

    if len(data) < frame_size:
        return np.array([np.sqrt(np.mean(data ** 2))], dtype=np.float32)

    shape = ((len(data) - frame_size) // hop_size + 1, frame_size)
    strides = (data.strides[0] * hop_size, data.strides[0])
    frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    rms_values = np.sqrt(np.mean(frames ** 2, axis=1))
    
    return rms_values

def rms_to_db[T: np.floating](rms_values: NDArray[T], ref: float = 1.0) -> NDArray[T]:
    """
    将RMS值转换为dB
    
    参数:
        rms_values: RMS值数组
        ref: 参考值 (1.0 表示归一化音频的 dBFS)
        min_db: 最小dB值 (防止静音时出现 -inf)
        
    返回:
        db_values: 对应的dB值
    """
    # 防止 log(0) 错误：给一个极小值 epsilon
    epsilon = 1e-10
    
    # 公式: 20 * log10(rms / ref)
    db_values = 20 * np.log10(np.maximum(rms_values, epsilon) / ref)
    
    return db_values

def split_by_silence(
    audio: Audio,
    *,
    frame_len: int = 20,
    hop_len: int = 10,
    silence_thresh_db: float = -35,
    ref: float = 1.0,
    min_silence_duration_ms: float = 300.0,
) -> List[Tuple[Audio, bool]]:
    """
    按静音切片，返回所有片段（含静音段）。

    参数:
        audio: 输入音频
        frame_len: 帧长度
        hop_len: 帧移
        silence_thresh_db: 静音阈值
        ref: RMS参考值
        min_silence_duration_ms: 最短静音持续时间。小于此值的静音段将被忽略，
                                 其前后两段语音会合并为一段。
        
    返回: List[Tuple[Audio, bool]]
        - Audio: 切片后的音频
        - bool:  True = 该段为静音, False = 该段为语音
    """
    data, sr = audio
    data = np.asarray(data)

    # 1. 复用 frame_rms + rms_to_db
    rms_values = frame_rms(audio, frame_len=frame_len, hop_len=hop_len)
    db_values  = rms_to_db(rms_values, ref=ref)
    is_silence = db_values < silence_thresh_db
    n_frames   = len(is_silence)

    # 2. 找静音区间起止帧（padding False 保证首尾闭合）
    padded = np.concatenate([[False], is_silence, [False]])
    diff   = np.diff(padded.astype(np.int8))
    sil_starts = np.where(diff == 1)[0]    # 静音开始帧
    sil_ends   = np.where(diff == -1)[0]   # 静音结束帧

    if len(sil_starts) > 0 and min_silence_duration_ms > 0:
            # 计算最短静音所需帧数 (向上取整，至少1帧)
            min_silence_frames = max(1, int(np.ceil(min_silence_duration_ms / hop_len)))
            
            # 计算每个静音段的持续帧数
            sil_lengths = sil_ends - sil_starts
            
            # 过滤出长度达标的静音段
            valid_mask = sil_lengths >= min_silence_frames
            sil_starts = sil_starts[valid_mask]
            sil_ends = sil_ends[valid_mask]

    # 3. 交错排列得到所有边界
    #    bounds = [0, ss0, se0, ss1, se1, ..., n_frames]
    #    偶数段索引 → 语音(False), 奇数段索引 → 静音(True)
    bounds = np.empty(len(sil_starts) + len(sil_ends) + 2, dtype=np.int64)
    bounds[0]  = 0
    bounds[-1] = n_frames
    if len(sil_starts) > 0:
        bounds[1:-1:2] = sil_starts
        bounds[2:-1:2] = sil_ends

    # 4. 帧索引 → 采样索引
    hop_size = int(round(hop_len * sr / 1000))
    sample_bounds = bounds * hop_size
    sample_bounds[-1] = len(data)  # 末尾对齐信号终点

    if bounds[-1] == bounds[-2]:
        sample_bounds[-2] = len(data) # 说明实际上没有这个段，这是靠 pad 出来的段

    results: List[Tuple[Audio, bool]] = []
    for j in range(len(sample_bounds) - 1):
        s = sample_bounds[j]
        e = sample_bounds[j + 1]
        seg_data = data[s:e]
        is_sil = (j % 2 == 1)           # 奇数段 = 静音

        if len(seg_data) == 0:
            continue

        results.append(((seg_data, sr), is_sil))

    return results

def print_segments_info(segments: List[Tuple[Audio, bool]]) -> None:
    """
    打印切片结果的时间戳和静音标注。
    
    参数:
        segments: split_by_silence 函数的返回结果
    """
    print(f"{'序号':<4} | {'开始时间 (s)':<12} | {'结束时间 (s)':<12} | {'时长 (s)':<10} | {'类型'}")
    print("-" * 60)
    
    current_time = 0.0
    
    for i, ((seg_data, sr), is_sil) in enumerate(segments):
        # 计算当前片段的时长
        duration = len(seg_data) / sr
        start_time = current_time
        end_time = start_time + duration
        
        # 更新时间游标
        current_time = end_time
        
        # 格式化输出
        tag = "[静音]" if is_sil else "[语音]"
        print(f"{i:<4} | {start_time:<12.3f} | {end_time:<12.3f} | {duration:<10.3f} | {tag}")