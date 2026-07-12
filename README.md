# [RvcPyInfer](https://github.com/TwoCreepers/RvcPyInfer)

[Rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的 ONNX 模型导出格式推理实现。  
支持在 Windows 上快速导出 ONNX 模型。

## 安装

本库尚未上传至 [PyPl](https://pypi.org) ，我们还在筹备这方面的工作。  
你可以直接 clone 该仓库然后使用如下命令构建 `.whl` 文件。
```shell
# 克隆该仓库
git clone https://github.com/TwoCreepers/RvcPyInfer.git
# 进入该仓库目录
cd RvcPyInfer
# 使用 uv 构建
uv build
```
不出意外的话，现在你需要的轮子文件应该出现在 `dist/` 下面了。  
当然了， `uv` 不是必须的，我们使用 `hatchling` 作为构建后端，你可以自己研究一下如何构建。

## 🚀快速使用
一个简单的示例：
```python
import RvcPyInfer

context = RvcContext() # 创建一个上下文

task = context.build_task(
    "./你的特征模型.onnx",
    "./你的 Rvc 生成器模型.onnx",
    48000 # 你的 Rvc 生成器模型的生成采样率,
    "./你要推理的源文件.wav"
    # 还有诸多可选参数等你来填
)
task.run_and_save(
    "./你要保存的结果.wav" # 请注意，它不会自动创建目录
)
```

## 关于特征模型
你可以在[这里](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)找到 `MoeSS` 使用的 `onnx` 特征模型，它们是通用的，通常情况下 `v2` 版本的生成器模型使用的是 `vec-768-layer-12.onnx`

## 关于推理引擎
你可能已经看到了那个未安装任何推理引擎的警告或者错误。  
请在安装命令后面添加 `[xxx]` 来安装可选依赖。  
例如：
```shell
pip install ./rvcpyinfer-0.1.0-py3-none-any[cpu]
```
目前我们支持一下推理引擎：

- `onnxruntime`
  - `[cpu]`
  - `[cuda]`
  - `[dml]`
- `openvino`
  - `[openvino]`

请注意 `onnxruntime` 三个可选依赖是互斥的，你只能选择其中的一个。  
关于 `openvino` 的 `IR` 格式，本库会转换并存档 `IR` 格式的模型以便在下一次加载时加速，该行为目前无法被禁用。

## 关于特征索引
原项目的 `.index` 文件可直接使用，无需转换。  
但你需要使用 `[index]` 来安装我们需要用于读取特征索引的依赖—— `faiss`。

## CLI
我们已经预制了一些 `CLI` 在库中了，并随库一并打包。  
以下是库内置的命令：  
- `rvc-infer`：用以快速推理模型而无需编写代码
- `rvc-model`：与原项目模型相关的命令

### rvc-model
该命令用于查看原项目的 `.pth` 模型的生成采样率，以及在 `Windows` 上快速导出 onnx 模型。

#### rvc-model show-sr
格式： `rvc-model show-sr -m <你的pth模型路径>`

#### rvc-model export
格式： `rvc-model export -m <你的pth模型路径> -t <你希望输出到哪> -r <rvc 原项目的根路径> --runtime <可选的导出用的 python 解释器路径，默认使用 rvc 原项目整合包自带的解释器>`

## 许可证
在文件头部或文件所在目录未有额外说明的情况下，本项目代码部分使用 [`MIT`](https://mit-license.org/) 许可证授权与你，非代码部分使用 `CC BY 4.0` 授权与你。  