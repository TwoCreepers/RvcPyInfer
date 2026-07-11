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

def rms_frame_match(source: Audio,
                    target: Audio,
                    frame_len: int = 20,
                    hop_len: int = 10,
                    mix: float = 1.0,
                    eps: float = 1e-8) -> Audio:
    """
    将 source 的逐帧 RMS 包络匹配到 target。

    参数:
        source:    源音频（将被增益调整）
        target:    目标音频（提取 RMS 包络的参考）
        frame_len: 帧长度(ms)
        hop_len:   帧移(ms)
        mix:       匹配系数，0.0=直接输出源音频，1.0=完全匹配目标RMS
        eps:       防除零，和检测 mix 是否为0

    返回:
        Audio: RMS 匹配后的音频
    """
    def _smooth_gain[T: np.floating](gain: NDArray[T], smooth_window: int = 15) -> NDArray[T]:
        """边缘延伸 padding 后卷积，常数增益卷积后仍为常数"""
        if smooth_window <= 1 or len(gain) <= 1:
            return gain
        kernel = np.ones(smooth_window) / smooth_window
        pad = smooth_window // 2
        padded = np.concatenate([
            np.full(pad, gain[0]),
            gain,
            np.full(pad, gain[-1])
        ])
        return np.convolve(padded, kernel, mode='valid').astype(gain.dtype)
    
    if mix < eps: # 因为太小了，几乎没有更改
        return source

    src_data, src_sr = source
    _, tgt_sr = target

    assert src_sr == tgt_sr, "采样率必须一致"

    N = len(src_data)

    # ---- 1. 逐帧 RMS ----
    src_rms = frame_rms(source, frame_len, hop_len)
    tgt_rms = frame_rms(target, frame_len, hop_len)
    assert len(src_rms) == len(tgt_rms), "帧数必须一致"

    # ---- 2. 逐帧增益 + 匹配系数混合 ----
    gain = tgt_rms / (src_rms + eps)
    gain = 1.0 + mix * (gain - 1.0)

    # ---- 3. 增益曲线平滑（边缘延伸 padding，无边界失真）----
    gain = _smooth_gain(gain)

    # ---- 4. 帧参数 ----
    frame_size = int(round(frame_len * src_sr / 1000))
    hop_size = int(round(hop_len * src_sr / 1000))
    n_frames = len(gain)

    # ---- 5. 逐帧增益 → 逐样本增益（overlap-add 归一化）----
    sample_gain = np.zeros(N, dtype=np.float64)
    norm = np.zeros(N, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, N)
        sample_gain[start:end] += gain[i]
        norm[start:end] += 1.0
    norm[norm == 0] = 1.0
    sample_gain /= norm

    # ---- 6. 尾部 ----
    if n_frames > 0:
        last_end = (n_frames - 1) * hop_size + frame_size
        if last_end < N:
            sample_gain[last_end:] = gain[-1]

    # ---- 7. 应用增益 ----
    output = src_data * sample_gain
    return (output.astype(np.float32), src_sr)

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
    min_silence_duration_ms: float = 800.0,
    max_transition_ms: float = 50
) -> List[Tuple[Audio, bool]]:
    """
    按静音切片，返回所有片段（含静音段）。

    参数:
        audio: 输入音频
        frame_len: 帧长度
        hop_len: 帧移
        silence_thresh_db: 静音阈值
        ref: RMS参考值
        min_silence_duration_ms: 最短静音持续时间。小于此值的静音段将被忽略。请注意，这不代表返回的静音段一定大于这个值
        max_transition_ms: 语音段的过渡时间，不得大于 0.5*min_silence_duration_ms
        
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

    # 5. 调整索引形成过渡
    max_transition_size = max_transition_ms * sr // 1000
    s = sample_bounds[1:-1:2]  
    e = sample_bounds[2::2]   
    sil_size = e - s
    voice_size = sample_bounds[1::2] - sample_bounds[0:-1:2]
    # 因为前面修正 pad 逻辑的时候会产生 0 长切片，这里不能对 0 长切片调整（因为那没有意义）
    voice_mask = voice_size > 0 
    start_mask = voice_mask[:-1]
    end_mask = voice_mask[1:]
    transition_size = np.minimum(max_transition_size, sil_size // 2)
    s[start_mask] = s[start_mask] + transition_size[start_mask]
    e[end_mask] = e[end_mask] - transition_size[end_mask]

    # 6. 切片前的合并
    # 6.1 处理零长
    is_sil = np.empty(len(sample_bounds) - 1, dtype=np.bool)
    is_sil[1::2] = True
    is_sil[0::2] = False
    index_diff = np.diff(sample_bounds)
    index_mask = index_diff > 0 # 0 长丢掉
    index_mask = np.pad(index_mask, [1, 0], mode="constant", constant_values=True)
    is_sil = is_sil[index_mask[1:]]
    sample_bounds = sample_bounds[index_mask]
    # 6.2 合并同类项
    sil_mask = is_sil[:-1] != is_sil[1:] 
    sil_mask = np.pad(sil_mask, [1, 1], mode="constant", constant_values=True)
    is_sil = is_sil[sil_mask[:-1]]
    sample_bounds = sample_bounds[sil_mask]

    # 7. 切片！
    return [((data[sample_bounds[i]:sample_bounds[i+1]], sr), is_sil[i]) for i in range(len(sample_bounds) - 1)]

def split_by_max_len_with_overlap(
        audio: Audio,
        *,
        max_len: int = 30,
        overlap_len: int = 5
    ) -> List[Audio]:
    data, sr = audio
    max_size = sr * max_len
    
    if overlap_len < 0 or overlap_len >= max_len or len(data) <= max_size:
        return [audio]
        
    overlap_size = sr * overlap_len
    step = max_size - overlap_size  # 每次向前推进的步长

    res: List[Audio] = []
    offset = 0
    
    while offset < len(data):
        end = min(offset + max_size, len(data))
        res.append((data[offset:end], sr))
        
        if end == len(data):
            break
            
        offset += step
        
    return res

def crossfade(
        audios: List[Audio],
        *,
        overlap_len: int = 5
    ) -> Audio:
    if len(audios) == 0:
        raise ValueError("音频列表为空")
    elif len(audios) == 1:
        return audios[0]

    _, sr = audios[0]
    datas = [d for d, _ in audios]
    overlap_size = sr * overlap_len
    fade_in = np.linspace(0.0, 1.0, overlap_size)
    fade_out = np.linspace(1.0, 0.0, overlap_size)

    data_list = []
    # 第一个
    data_list.append(datas[0][:-overlap_size])
    data_list.append(datas[0][-overlap_size:] * fade_out + datas[1][:overlap_size] * fade_in)

    # 中间
    for i in range(1, len(datas) - 1):
        curr = datas[i]
        next = datas[i + 1]
        data_list.append(curr[overlap_size:-overlap_size])
        data_list.append(curr[-overlap_size:] * fade_out + next[:overlap_size] * fade_in)

    # 最后一个
    data_list.append(datas[-1][overlap_size:])

    return np.concatenate(data_list), sr

def copy_audio(audios: List[Audio]) -> List[Audio]:
    for i in range(1, len(audios)):
        audios[i] = (audios[i][0].copy(), audios[i][1])
    return audios

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