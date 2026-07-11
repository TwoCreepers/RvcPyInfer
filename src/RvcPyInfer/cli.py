import argparse
import sys

from .InferProviders import InferProviders
from .RvcContext import RvcContext

def str_to_provider(provider_str: str) -> InferProviders:
    """将命令行字符串映射为 InferProviders 枚举"""
    if provider_str == "default":
        return InferProviders.default()
    elif provider_str == "cuda":
        return InferProviders.CUDA
    elif provider_str == "dml":
        return InferProviders.DML
    elif provider_str == "ort_cpu":
        return InferProviders.ORT_CPU
    elif provider_str == "openvino_auto":
        return InferProviders.OPENVINO_AUTO
    elif provider_str == "openvino_cpu":
        return InferProviders.OPENVINO_CPU
    elif provider_str == "openvino_gpu":
        return InferProviders.OPENVINO_GPU
    else:
        raise ValueError(f"未知的 Provider: {provider_str}")

def main():
    parser = argparse.ArgumentParser(description="RVC 语音转换命令行工具")
    
    # --- 必填参数 ---
    parser.add_argument("--vec_model", type=str, required=True, help="ContentVec 特征提取模型路径 (.onnx)")
    parser.add_argument("--gen_model", type=str, required=True, help="RVC 生成模型路径 (.onnx)")
    parser.add_argument("--gen_model_sr", type=int, required=True, help="生成模型采样率 (如 40000, 48000)")
    parser.add_argument("-i", "--inputs", nargs="+", type=str, required=True, help="输入音频文件路径 (可传入多个)")
    parser.add_argument("-o", "--outputs", nargs="+", type=str, required=True, help="输出音频文件路径 (数量必须与输入一致)")
    
    # --- 推理引擎 ---
    parser.add_argument("--provider", type=str, default="default", 
                        choices=["default", "cuda", "dml", "ort_cpu", "openvino_auto", "openvino_cpu", "openvino_gpu"],
                        help="推理后端 (默认自动检测)")
    
    # --- 基础设置 ---
    parser.add_argument("--sid", type=int, default=0, help="说话人 ID")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    
    # --- 特征索引 ---
    parser.add_argument("--index_path", type=str, default=None, help="特征索引文件路径")
    parser.add_argument("--index_rate", type=float, default=0.33, help="特征索引比率 (0-1)")
    parser.add_argument("--index_k", type=int, default=8, help="检索的最近特征数量")
    
    # --- f0 提取 ---
    parser.add_argument("--f0_method", type=str, default="dio", help="f0 提取算法 (如 dio, pm, rmvpe, fcpe 等)")
    parser.add_argument("--pitch", type=float, default=0, help="升降调 (半音)")
    parser.add_argument("--f0_min", type=float, default=50, help="最低 f0")
    parser.add_argument("--f0_max", type=float, default=1100, help="最高 f0")
    
    # --- 强制切片与交叉淡化 ---
    parser.add_argument("--slice_max_len", type=int, default=30, help="最大切片长度 (秒)")
    parser.add_argument("--slice_overlap_len", type=int, default=5, help="切片交叉淡化长度 (秒)")
    
    # --- 静音切片 ---
    parser.add_argument("--silence_thresh_db", type=float, default=-40, help="静音阈值")
    parser.add_argument("--silence_min_duration", type=float, default=800.0, help="最小静音时长")
    parser.add_argument("--silence_max_transition", type=float, default=100.0, help="最大过渡时长")
    
    args = parser.parse_args()

    # 校验输入输出数量
    if len(args.inputs) != len(args.outputs):
        print("错误: 输入音频数量与输出音频数量不一致！", file=sys.stderr)
        sys.exit(1)

    try:
        # 1. 初始化推理 Provider 和 Context
        print(f"正在初始化推理环境，使用 Provider: {args.provider} ...")
        providers = str_to_provider(args.provider)
        context = RvcContext(providers=providers)
        
        # 2. 构建 InferTask
        print("正在构建推理任务...")
        task = context.build_task(
            vec_model=args.vec_model,
            gen_model=args.gen_model,
            gen_model_sr=args.gen_model_sr,
            *args.inputs,
            sid=args.sid,
            seed=args.seed,
            index_path=args.index_path,
            index_rate=args.index_rate,
            index_k=args.index_k,
            f0extract_algorithm=args.f0_method,
            f0_up_semitone=args.pitch,
            f0_min=args.f0_min,
            f0_max=args.f0_max,
            slice_max_len=args.slice_max_len,
            slice_overlap_len=args.slice_overlap_len,
            silence_thresh_db=args.silence_thresh_db,
            silence_min_silence_duration_ms=args.silence_min_duration,
            silence_max_transition_ms=args.silence_max_transition,
        )
        
        # 3. 执行推理并保存
        print("开始推理，请稍候...")
        task.run_and_save(*args.outputs)
        print(f"推理完成！音频已保存至: {', '.join(args.outputs)}")
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
