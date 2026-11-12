from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from pathlib import Path
import torch

lora_adapter_path = Path(__file__).parent / "d26_lora_adapter"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)
loaded_model = PeftModel.from_pretrained(
    base_model,
    str(lora_adapter_path)
)

question = "hugging face是一个什么类型的网站?"
message = [{"role": "user","content": question}]

tokenized_question = tokenizer.apply_chat_template(
    message,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict=True
).to(loaded_model.device)
# tokenized_question = {"input_ids": ..., "attention_mask": ...}
input_ids = tokenized_question["input_ids"]
print(input_ids.shape)
print(tokenizer.decode(input_ids[0]))

loaded_model.eval()
with torch.no_grad():
    generated_ids = loaded_model.generate(**tokenized_question,max_new_tokens = 100,do_sample = False)
    # 返回的是一个二维tensor 且包含了原始输入token + 新生成的token 即([B,T])
    new_tokens = generated_ids[0][input_ids.shape[1]:]
    # generated_ids[0] 取出 Batch 中的第一个样本 也即 ([T]) 这里是tensor
    # input_ids.shape[1] 则是取shape(B,T) 里面的T 这里是长度
    answer = tokenizer.decode(new_tokens,skip_special_tokens=True)
    print("模型回答：", answer)