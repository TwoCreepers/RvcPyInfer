const_tips = """# ----------------------------------------------------------------
# 这是自动生成的临时脚本，若您看见工具未能正常删除的话，您可以手动删除它
# 您当然可以尝试运行了，甚至保存下来，脚本引用的是绝对路径
# ----------------------------------------------------------------

"""

import re
import os
import json
import uuid
import random
import hashlib
import subprocess
from typing import List, Dict
from datetime import datetime
from importlib.resources import files

from ...path_utils import path
from ...type_alist import PathLike

class Exporter:
    def __init__(self, root: PathLike, model: PathLike, export_target: PathLike, runtime: PathLike | None = None) -> None:
        self.root = path(root).resolve()
        self.model = path(model).resolve()
        self.target = path(export_target).resolve()
        if runtime is None:
            self.runtime = self.root / "runtime" / "python.exe"
        else:
            self.runtime = path(runtime).resolve()
        self.parse_prefix = self.build_parse_prefix()
        template = files("RvcPyInfer.resource.template").joinpath("export_onnx.py").read_text(encoding="utf-8")
        self.template_digest = hashlib.sha256(template.encode("utf-8"))
        self.script = self.format_template(template)

    @staticmethod
    def build_parse_prefix() -> str:
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        tmp: List[str] = []
        for _ in range(8):
            tmp.append(alphabet[random.randint(0, len(alphabet) - 1)])
        return "".join(tmp)

    def format_template(self, template: str) -> str:
        model = str(self.model).replace("\\", "/")
        target = str(self.target).replace("\\", "/")
        root = str(self.root).replace("\\", "/")
        template = template.replace("/./..{code.InsertionPoint.0}.././", f'{model}')
        template = template.replace("/./..{code.InsertionPoint.1}.././", target)
        template = template.replace("/./..{code.InsertionPoint.2}.././", self.parse_prefix)
        template = template.replace("/./..{code.InsertionPoint.3}.././", root)

        args_digest = hashlib.sha256(f"model: {model}, target: {target}, root: {root}".encode("utf-8"))

        gen_info = [
            "\n"
            "# ----------------------------------------------------------------\n"
            "# -------- 脚本元信息 --------\n"
            f"# 该脚本生成于: {datetime.now()}\n",
            f"# 模板摘要: {self.template_digest.hexdigest()}\n"
            f"# 非随机参数摘要: {args_digest.hexdigest()}\n"
            f"# 模型: {model}\n"
            f"# 目标: {target}\n"
            f"# Rvc 项目根目录: {root}\n"
            f"# 解析前缀: {self.parse_prefix}\n"
            f"# 系统执行所用的 runtime: {self.runtime}\n"
            "# ----------------------------------------------------------------\n"
            "\n"
        ]

        gen_info = ''.join(gen_info)

        return const_tips + gen_info + template
    
    def parse_stdout(self, stdout: str) -> Dict[str, str | int]:
        lines = stdout.split('\n')
        info = {}
        for line in lines:
            line = line.strip()
            if line.startswith(self.parse_prefix):
                match = re.match(r'^\/\.\/\.\.(.*)\.\.\/\.\/$', line[len(self.parse_prefix):])
                if match is None:
                    continue
                else:
                    json_info = match.group(1)
                    info |= json.loads(json_info)
        return info
    
    def export(self) -> None:
        tmp_py_path = (self.root / str(uuid.uuid4())).with_suffix(".py")
        with open(tmp_py_path, mode="+w", encoding="utf-8") as f:
            f.write(self.script)
        try:
            result = subprocess.run(
                args=[str(self.runtime), str(tmp_py_path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.root
            )
        finally:
            try:
                if tmp_py_path.exists():
                    os.remove(tmp_py_path)
            except Exception:
                ...
        info = self.parse_stdout(result.stdout)
        sr = info.get("sr")
        if sr is not None:
            print(f"模型采样率: {sr}")
        
        if result.returncode == 0:
            print("导出成功！")
            return
        else:
            msg = info.get("msg")
            if msg is None:
                print("导出失败")
            else:
                if isinstance(msg, str):
                    msg = msg.replace("\\\"", "\"").replace("\\n", "\n")
                print(f"导出失败: {msg}")
            print("-------- stdout --------")
            print(result.stdout)
            print("------------------------")
            print()
            print("-------- stderr --------")
            print(result.stderr)
            print("------------------------")
            print()
            print(f"return code: {result.returncode}")