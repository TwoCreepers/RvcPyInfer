from .onnx.export.cli import main as export_main
from .cli import main as infer_main

if __name__ == "__main__":
    infer_main()