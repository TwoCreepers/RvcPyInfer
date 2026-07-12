# [RvcPyInfer](https://github.com/TwoCreepers/RvcPyInfer)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://mit-license.org/)  
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)  
![ONNX](https://img.shields.io/badge/ONNX-supported-blue)

[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的 ONNX 模型导出格式推理实现。  

- 支持原 RVC 项目的 ONNX 模型推理。  
- 跨平台支持多种推理引擎（CPU, CUDA, DirectML, OpenVINO）。  
- 提供 Windows 下一键导出 ONNX 模型的 CLI 工具。  
- 兼容原项目的 `.index` 特征索引文件。  

## Python 版本需求
本项目要求 Python 版本 `>= 3.13`。由于项目代码中使用了 `Python 3.12+` 引入的 `TypeParam` 等现代语法特性，因此无法向下兼容更低版本的 Python。  
~~并且推理真的很难测试欸。~~

## 安装
本库尚未上传至 [PyPI](https://pypi.org) ，我们还在筹备这方面的工作。  
你可以直接 clone 该仓库然后使用如下命令构建 `.whl` 文件。
```shell
# 克隆该仓库
git clone https://github.com/TwoCreepers/RvcPyInfer.git
# 进入该仓库目录
cd RvcPyInfer
# 使用 uv 构建
uv build
# 安装生成的 wheel 包 (以实际文件名为准)
pip install ./dist/rvcpyinfer-0.1.0-py3-none-any.whl
```
不出意外的话，现在你需要的轮子文件应该出现在 `dist/` 下面了。  
这里出现的 [`uv`](https://github.com/astral-sh/uv) 是一个使用 `Rust` 编写的 `Python` 项目管理工具。  
它可以为你剩下非常多在依赖管理方面的麻烦，就比如上面的 `uv build` 它会自动为你生成虚拟环境并安装好依赖。  
当然了， [`uv`](https://github.com/astral-sh/uv) 不是必须的，我们使用 [`hatchling`](https://pypi.org/project/hatchling) 作为构建后端，你可以自己研究一下如何构建。  

或者你也可以直接把当前项目作为库安装，例如：
```shell
# 假设你已经在仓库目录里了
pip install .
# 如果需要安装 CPU 引擎和特征索引支持的话是
pip install ".[cpu,index]"
```


## 关于特征模型
你可以在[这里](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)找到 `MoeSS` 使用的 `onnx` 特征模型。它们是通用的。  
**通常情况下最常见的 `v2` 版本的生成器模型使用的是 `vec-768-layer-12.onnx`**

## 关于推理引擎
你可能已经看到了那个未安装任何推理引擎的警告或者错误。  
请在安装命令后面添加 `[xxx]` 来安装可选依赖。  
例如：
```shell
# 假设你已经在 dist/ 目录下构建好了 whl 文件
pip install "./dist/rvcpyinfer-0.1.0-py3-none-any[cpu]"
```
目前我们支持以下推理引擎：

- `onnxruntime`
  - `[cpu]`：ONNX Runtime (CPU)，通用
  - `[cuda]`：ONNX Runtime (CUDA)，Nvidia GPU 加速
  - `[dml]`：ONNX Runtime (DML)，Windows DirectML 加速
- `openvino`
  - `[openvino]`：OpenVINO，Intel 硬件加速

请注意 `onnxruntime` 的三个可选依赖是互斥的，你只能选择其中的一个。  
关于 `openvino` 的 `IR` 格式，本库会转换 `IR` 格式并存档在 `.onnx` 同目录下的同名文件中，以便在下一次加载时加速，该行为目前无法被禁用。

## 关于特征索引
原项目的 `.index` 文件可直接使用，无需转换。  
但你需要使用 `[index]` 来安装我们需要用于读取特征索引的依赖—— `faiss`。  

## 快速使用
一个简单的示例：
```python
import RvcPyInfer

context = RvcPyInfer.RvcContext() # 创建一个上下文

task = context.build_task(
    "./你的特征模型.onnx",
    "./你的 Rvc 生成器模型.onnx",
    48000, # 你的 Rvc 生成器模型的生成采样率
    "./你要推理的源文件.wav", # 可以填多个源文件
    # 后续可选参数需显式写明名称，例如: 
    sid=1,    # 说话者 id，一般情况下就是 0
    seed=4321 # 噪声种子，默认 1234
)
task.run_and_save(
    "./你要保存的结果.wav" # 请注意，它不会自动创建目录
)
```

## CLI
我们已经预制了一些 `CLI` 在库中了，并随库一并打包。  
以下是库内置的命令：  
- `rvc-infer`：用以快速推理模型而无需编写代码
- `rvc-model`：与原项目模型相关的命令

### rvc-infer
最简格式： `rvc-infer --vec-model <你的特征模型路径> --gen-model <你的生成器模型路径> --gen-model-sr <你的生成器模型的输出采样率> -i <输入音频文件路径> -o <输出音频文件路径>`  
**Tips：可出现多个 -i 和 -o，只要它们的数量一致，多个 -i 和 -o 按出现顺序一一对应。**

### rvc-model
该命令用于查看原项目的 `.pth` 模型的生成采样率，以及在 `Windows` 上快速导出 onnx 模型。

#### rvc-model show-sr
格式： `rvc-model show-sr -m <你的pth模型路径>`

#### rvc-model export
**⚠️注意：你必须安装源项目的整合包或至少有一个能运行源项目的环境以便 `PyTorch` 导出 `.onnx` 模型。**  
它并不依赖项目内置的 `tools/export_onnx.py` 我们有自己的方法。  
~~事实上原项目的 export_onnx.py 甚至没有做 v2 版本的支持~~  
格式： `rvc-model export -m <你的pth模型路径> -t <你希望输出到哪> -r <rvc 原项目的根路径> --runtime <可选的导出用的 python 解释器路径，默认使用 rvc 原项目整合包自带的解释器>`  

## 未来的计划
- [ ] Github Action 可重现性构建流水线  
- [ ] 发布至 PyPI  

## 许可证
在文件头部或文件所在目录未有额外说明的情况下，本项目代码部分使用 [`MIT`](https://mit-license.org/) 许可证授权与你，非代码部分使用 `CC BY 4.0` 授权与你。  

## 🎉鸣谢

- [原项目组的所有成员](https://github.com/RVC-Project)  
没有TA们的付出就没有 RVC 模型，该项目也不会出现  
- [提前打好特征模型 ONNX 的 MoeSS](https://github.com/Miuzarte/MoeSS)  
省去了诸多导出特征模型的麻烦  
- [帮了作者很多的 GLM AI](https://chatglm.cn)  
省去了很多查资料和写 CLI 的时间  