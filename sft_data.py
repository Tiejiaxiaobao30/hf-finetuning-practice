from transformers import AutoTokenizer
from pathlib import Path
from datasets import load_dataset

file_path = Path(__file__).parent/"train.jsonl"
dataset = load_dataset(
    "json",
    data_files = str(file_path),
    split="train"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)

#小练习
input_ids = [11, 12, 13, 21, 22, 0, 0]
prompt_length = 3
real_length = 5
labels = input_ids.copy()
labels[:prompt_length]= [-100]*3  # 列表切片，赋值时需要提供一个可迭代对象
labels[real_length:]= [-100]*2
print(labels)

instruction = "什么是机器学习？"
output = "机器学习是让计算机从数据中学习规律的方法。"
prompt_messages = [{"role": "user", "content": instruction}]
full_messages = [{"role": "user", "content": instruction},{"role": "assistant", "content": output}]

#对话消息 → 格式化 → Token ID → Tensor → 字典式编码结果
tokenized_prompt = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict=True
)
tokenized_full = tokenizer.apply_chat_template(
    full_messages,
    tokenize = True,
    add_generation_prompt = False,
    return_tensors = "pt",
    return_dict=True
)
prompt_ids = tokenized_prompt["input_ids"][0].tolist() #.tolist()将 Tensor 转换为普通 Python 列表
full_ids = tokenized_full["input_ids"][0].tolist()
print(len(prompt_ids))
print(len(full_ids))
print(full_ids[:len(prompt_ids)] == prompt_ids)

labels = full_ids.copy()
labels[:len(prompt_ids)] = [-100]*len(prompt_ids)
print(labels)

# 得到 Token ID 后，手动padding
max_length = 64
tokenizer.pad_token = tokenizer.eos_token
pad_count = max_length - len(full_ids)
padded_input_ids = full_ids + [tokenizer.pad_token_id] * pad_count
attention_mask = tokenized_full["attention_mask"][0].tolist() + [0] * pad_count
padded_labels = labels + [-100]*pad_count
print(len(padded_input_ids))
print(len(attention_mask))
print(len(padded_labels))
print(padded_labels.count(-100))

# 封装成函数（今日最终目标）
def prepare_sft_example(example):
    instruction = example["instruction"]
    output = example["output"]
    prompt_messages = [{"role": "user", "content": instruction}]
    full_messages = [{"role": "user", "content": instruction},{"role": "assistant", "content": output}]
    tokenized_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict=True
    )
    tokenized_full = tokenizer.apply_chat_template(
        full_messages,
        tokenize = True,
        add_generation_prompt = False,
        return_tensors = "pt",
        return_dict=True
    )
    prompt_ids = tokenized_prompt["input_ids"][0].tolist()
    full_ids = tokenized_full["input_ids"][0].tolist()
    labels = full_ids.copy()
    labels[:len(prompt_ids)] = [-100]*len(prompt_ids)
    max_length = 64
    tokenizer.pad_token = tokenizer.eos_token
    if len(full_ids) > max_length:
        raise ValueError("样本超过最大长度")
    pad_count = max_length - len(full_ids)
    padded_input_ids = full_ids + [tokenizer.pad_token_id] * pad_count
    attention_mask = tokenized_full["attention_mask"][0].tolist() + [0] * pad_count
    padded_labels = labels + [-100]*pad_count
    return {
        "input_ids":padded_input_ids,
        "attention_mask":attention_mask,
        "labels":padded_labels
    }
sft_dataset = dataset.map(prepare_sft_example)
print(sft_dataset.column_names)

for example in sft_dataset:
    print(
        len(example["input_ids"]),
        len(example["attention_mask"]),
        len(example["labels"]),
        example["labels"].count(-100)
    )