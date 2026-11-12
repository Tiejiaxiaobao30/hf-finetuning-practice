# Hugging Face 微调练习

记录学习 Hugging Face Transformers、Datasets 和 PEFT 时的 Python 练习，以 Qwen2.5-0.5B-Instruct 为例。

- `transformers_inference.py`：Tokenizer、对话模板与基础推理。
- `datasetss.py`、`sft_data.py`：数据处理、回答部分监督与 Padding 屏蔽。
- `peft_lora.py`、`load_adapter.py`：LoRA 参数、手动训练与 Adapter 加载。
- `train.py`、`checkingpoint_resume.py`：Trainer SFT、保存检查点与恢复训练。
- `infer.py`：加载 SFT Adapter 生成回答。

## 使用说明

需要 Python、PyTorch、transformers、datasets、peft 和 accelerate，并提前准备模型本地缓存。依赖版本尚未整理为可复现环境。

仓库附带 `train.jsonl`，包含 3 条用于演示的问答文本，每行包含 `instruction` 和 `output` 两个字符串字段。模型、Adapter 权重和检查点不随仓库上传。

运行前请核对路径：基础推理脚本使用个人本地模型路径；`peft_lora.py` 将 Adapter 保存到当前工作目录，`load_adapter.py` 从脚本旁的 `lora_adapter` 加载；`infer.py` 需要 `train.py` 生成的 `d26_lora_adapter`；恢复训练需要 `huggingface_outputs/checkpoint-1`。

这是学习过程中的源码记录，保留原练习写法。本次上传仅做源码审查与语法检查，未重新执行训练或验证模型效果。
