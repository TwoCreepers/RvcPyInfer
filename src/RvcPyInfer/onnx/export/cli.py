import argparse
import sys
from pathlib import Path

from .Exporter import Exporter
from .direct_read_sr import direct_read_sr

def export_command(args) -> None:
    """处理 export 子命令的具体逻辑"""
    root_path = Path(args.root).resolve()
    model_path = Path(args.model).resolve()
    target_path = Path(args.target).resolve()
    runtime_path = Path(args.runtime).resolve() if args.runtime else None

    # 前置检查
    if not root_path.is_dir():
        print(f"[错误] 根目录不存在或不是文件夹: {root_path}")
        sys.exit(1)
        
    if not model_path.is_file():
        print(f"[错误] 找不到模型文件: {model_path}")
        sys.exit(1)

    # 确保输出目录存在
    if not target_path.parent.exists():
        print(f"[提示] 创建输出目录: {target_path.parent}")
        target_path.parent.mkdir(parents=True, exist_ok=True)

    # 执行导出
    try:
        print(">>> 开始初始化导出器...")
        exporter = Exporter(
            root=root_path,
            model=model_path,
            export_target=target_path,
            runtime=runtime_path
        )
        
        print(">>> 正在执行导出流程，请耐心等待...")
        exporter.export()
        
    except FileNotFoundError as e:
        print(f"\n[错误] 找不到必要的文件: {e}")
        print("请检查模板文件 (export_onnx.py) 是否存在于 resource/template 目录下，或运行时路径是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 导出过程中发生未预期的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_sr_command(args) -> None:
    """处理 show-sr 子命令的具体逻辑"""
    model_path = Path(args.model).resolve()

    if not model_path.is_file():
        print(f"[错误] 找不到模型文件: {model_path}")
        sys.exit(1)

    sr = direct_read_sr(model_path)
    print(f"模型采样率是: {sr}")

def main() -> None:
    # 主解析器
    parser = argparse.ArgumentParser(
        description="RvcPyInfer 命令行工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="可用的子命令", required=True)

    # 注册 export 子命令
    export_parser = subparsers.add_parser(
        "export", 
        help="导出 RVC 模型为 ONNX 格式",
        description="将 RVC .pth 模型导出为 ONNX 格式"
    )
    # export 子命令的参数
    export_parser.add_argument("-r", "--root", type=str, required=True, help="RVC 项目根目录路径")
    export_parser.add_argument("-m", "--model", type=str, required=True, help="输入的 RVC 模型路径")
    export_parser.add_argument("-t", "--target", type=str, required=True, help="导出的 ONNX 模型目标路径")
    export_parser.add_argument("--runtime", type=str, default=None, help="自定义 Python 运行时路径 (默认: <root>/runtime/python.exe)")
    # 绑定执行函数
    export_parser.set_defaults(func=export_command)

    # 注册 show-sr 子命令
    show_sr_parser = subparsers.add_parser(
        "show-sr",
        help="直接查看 RVC 模型的输出采样率",
        description="查看 RVC 模型的输出采样率"
    )
    show_sr_parser.add_argument("-m", "--model", type=str, required=True, help="输入的 RVC 模型路径")
    show_sr_parser.set_defaults(func=show_sr_command)

    # 解析参数
    args = parser.parse_args()

    # 调用对应的处理函数
    args.func(args)

if __name__ == "__main__":
    main()