import torch 
from transformers import AutoTokenizer,AutoModelForCausalLM
#如果写 import peft 下面调用工具时需要写 config = peft.LoraConfig(r=8)
#如果写 from peft import LoraConfig 下面调用工具时需要写 config = LoraConfig(r=8)
#但是这两种的前提 都是已经pip install peft

model_path = r"C:\Users\冯铁\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
#指定本地模型目录 之前在本地下载过的模型
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only = True  #在写了这行的情况下 前面那个model_path也可以直接写模型id 
)
# tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct") 这种写法则是直接下载 只用去查询模型id即可

text = "我正在学习hugging face。"
token_id = tokenizer.encode(text)
decoded = tokenizer.decode(token_id)
print(token_id)
print(decoded)
print(tokenizer.convert_ids_to_tokens(token_id)) # ID 分别对应什么 token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only = True #在写了这行的情况下 前面那个model_path也可以直接写模型id 
)
# model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct") 这种写法则是直接下载 只用去查询模型id即可
print(type(model))
print(model.device)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "请用一句话介绍机器学习"},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, 
    tokenize=True,
    add_generation_prompt=True,  # 在末尾添加助手开始回答的标记
    return_tensors="pt",         # 将 token ID 返回为 PyTorch Tensor
    return_dict=True,            # 返回一个包含多个字段的字典式对象
).to(model.device)

model_inputs = tokenizer(["请用一句话介绍机器学习"], return_tensors="pt").to(model.device)  
# model_inputs 不是单个 Tensor，而是一个包含 input_ids、attention_mask 等信息的字典式对象

generated_ids1 = model.generate(**model_inputs,max_new_tokens = 30)
generated_ids2 = model.generate(**tokenized_chat,max_new_tokens = 30) 
# **的作用是把对象中的键值对展开为关键字参数
# 或者也可以写成 model.generate(input_ids=model_inputs["input_ids"],attention_mask=model_inputs["attention_mask"],max_new_tokens=30)
print(tokenizer.decode(generated_ids1[0],skip_special_tokens=True))
print(tokenizer.decode(generated_ids2[0],skip_special_tokens=False))
# 把生成的 ID 转换回文本 这里结果是一批解码后的字符串（列表） [0]是取第一条完整的生成文本