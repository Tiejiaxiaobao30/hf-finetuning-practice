from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model,TaskType,peft_model
import torch 

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only = True)

# 寻找 LoRA 目标模块
for name,module in model.named_modules():
    if "q_proj" in name or "v_proj" in name:   # or 两边需要分别写出完整的判断条件
        print(name)

# W'= W + (lora_alpha/r) * BA  A随机初始化，B初始化为全零矩阵，刚开始 BA=0
# r 决定 LoRA 分支用多大的低秩空间学习，alpha 决定学到的更新以多大的缩放系数作用到原模型上
lora_config = LoraConfig(
    r=8,                                  # LoRA 低秩矩阵的秩
    lora_alpha=16,                        # 控制 LoRA 更新的缩放
    target_modules=["q_proj","v_proj"],   # 指定插入 LoRA 的层
    lora_dropout=0.05,                    # LoRA 分支的 Dropout 比例
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model,lora_config)  # 把 LoRA Adapter 添加到原始模型中
peft_model.print_trainable_parameters()         # 获取所有参数和可训练参数信息

trainable_count = 0
frozen_count = 0
for name, param in peft_model.named_parameters(): 
    # 打印一个 name 和 param.shape 
    # 可能得到 model.layers.0.self_attn.q_proj.weight  torch.Size([896, 896]) 这种
    if param.requires_grad:
        trainable_count += param.numel()   # 如果写 +=1 统计的是parameter对象的数量  .numel()自动计算出参数量了
        if "lora_" not in name:
            print("发现非 LoRA 可训练参数：", name)
    else:
        frozen_count += param.numel()
print(f"可训练参数：{trainable_count}")
print(f"冻结参数：{frozen_count}")

# 准备训练数据
text ="机器学习。"
device = next(peft_model.parameters()).device  # 从模型的第一个参数中获取真实设备
model_inputs = tokenizer(
    text,
    return_tensors = "pt"
).to(device) 
labels = model_inputs["input_ids"].clone()

trainable_params = [p for p in peft_model.parameters()if p.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1e-4,
    weight_decay=0.0
)

#第一次训练：Forward → Loss → Backward
peft_model.train()  # 将模型切换到训练模式，例如启用 LoRA 分支中的 Dropout
optimizer.zero_grad(set_to_none=True)
outputs = peft_model(**model_inputs,labels = labels) 
#这里额外输入了labels，hugging face会自动进行前向传播和损失计算，不需要额外定义损失函数了
loss = outputs.loss
print("第一次 Loss：", loss.item())
loss.backward()
optimizer.step()

# 第二次训练：验证 A 开始获得梯度
optimizer.zero_grad(set_to_none=True)
outputs = peft_model(**model_inputs,labels=labels)
loss = outputs.loss
print("第二次 Loss：", loss.item())
loss.backward()
optimizer.step()

peft_model.save_pretrained("./lora_adapter")


