from sft_data import prepare_sft_example
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM,Trainer,TrainingArguments
from peft import LoraConfig,TaskType,get_peft_model
import torch

file_path = Path(__file__).parent/"train.jsonl"

dataset = load_dataset(
    "json",
    data_files = str(file_path),
    split="train"
)
sft_train_dataset = dataset.map(prepare_sft_example)
# map() 默认会在原数据集上新增或更新字段，并不会自动删除旧字段
# 即['instruction', 'output', 'input_ids', 'attention_mask', 'labels']
splits = sft_train_dataset.train_test_split(test_size=1,seed=42)
#本次只将上面的train_dataset留出1条来作为验证集 最终返回一个DatasetDict
train_dataset = splits["train"]
eval_dataset = splits["test"]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)
loaded_model = get_peft_model(model,lora_config)

training_args = TrainingArguments(
    output_dir = str(Path(__file__).resolve().parent / "huggingface_outputs"),     # 固定在脚本旁，避免随启动目录变化
    per_device_train_batch_size = 1,          # 训练batch数
    num_train_epochs = 1,
    learning_rate = 1e-4,                     # 初始学习率
    logging_steps = 1,                        # 每隔多少个优化器 Step 记录一次日志
    report_to = "none",                       # 使用哪些外部日志集成
    save_strategy = "steps",                  # 训练过程中什么时候自动保存 Checkpoint
    save_steps = 1,
    save_total_limit=2                        # 最多保留两个 Checkpoint
)

trainer = Trainer(
    model = loaded_model,
    args = training_args,
    train_dataset = train_dataset.select_columns(["input_ids","attention_mask","labels"]),
    eval_dataset = eval_dataset.select_columns(["input_ids","attention_mask","labels"])
)

train_loader = trainer.get_train_dataloader()
# 根据 TrainingArguments 和 train_dataset，自动构建用于训练的 PyTorch DataLoader
batch = next(iter(train_loader))
print(batch.keys())
for name, tensor in batch.items():
    print(name, tensor.shape)
print("有效监督 Token 数：", (batch["labels"] != -100).sum().item())

checkpoint_path = Path(training_args.output_dir) / "checkpoint-1"

train_result = trainer.train(resume_from_checkpoint=str(checkpoint_path))
print("恢复训练结束")
print("Global Steps:", train_result.global_step)
print("Train Loss:", train_result.training_loss)
