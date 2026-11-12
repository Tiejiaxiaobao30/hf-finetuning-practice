from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from pathlib import Path
import torch

lora_adapter_path = Path(__file__).parent / "lora_adapter"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)
loaded_model = PeftModel.from_pretrained(
    base_model,
    str(lora_adapter_path)
)

print(type(loaded_model))
print(loaded_model.peft_config)

text = "机器学习。"
loaded_model.eval()

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(next(loaded_model.parameters()).device)

with torch.no_grad():
    outputs = loaded_model(**inputs)

print(outputs.logits.shape)