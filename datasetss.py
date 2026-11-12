from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)
tokenizer.pad_token = tokenizer.eos_token   
# 把已有的 EOS Token 指定为 Padding Token(暂时这样写) 
# 因为Qwen Tokenizer 没有预先设置独立的 pad_token

file_path = Path(__file__).parent/"train.jsonl"
dataset = load_dataset(
    "json",
    data_files =str(file_path), 
    split="train"              # 直接获得 Dataset 对象
)
print(len(dataset))
print(dataset[0])

def format_example(example):
    data = {}
    instruction = example["instruction"]
    output = example["output"]
    data["text"] = f"问题:{instruction} 回答：{output}"
    return data

new_dataset = dataset.map(format_example) # dataset.map() 会逐条处理 dataset 中的样本
print(new_dataset)

def tokenizer_example(examples):
    text = examples["text"]
    encoded = tokenizer(
        text,
        padding = "max_length",
        truncation = True,
        max_length = 64   # padding、truncation 和 max_length 都是 Tokenizer 的参数
    )
    return encoded

tokenized_dataset = new_dataset.map(tokenizer_example,batched=True,batch_size=2)

print(tokenized_dataset.column_names)
print(tokenized_dataset[0]["input_ids"])
print(tokenized_dataset[0]["attention_mask"])

for example in tokenized_dataset:
    print(
        "input_ids长度：", len(example["input_ids"]),
        "attention_mask长度：", len(example["attention_mask"]),
        "Padding数量：", example["attention_mask"].count(0)
    )
