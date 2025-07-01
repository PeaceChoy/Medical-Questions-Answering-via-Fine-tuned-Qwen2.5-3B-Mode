from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import json

# 设置环境变量以优化内存使用
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置模型保存路径
model_save_path = "models/Qwen2.5-3B-Instruct"
output_dir = "MedQA/output/Qwen2.5-3B-3th"

# 确保保存路径存在
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

model_name = "Qwen/Qwen2.5-3B-Instruct"

# 检查模型是否已经下载
if not os.path.exists(os.path.join(model_save_path, "config.json")):
    print("正在下载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={'': torch.cuda.current_device()},  # 确保模型在正确的设备上
        load_in_8bit=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 保存模型和分词器
    print("正在保存模型到本地...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
else:
    print("从本地加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_save_path,
        torch_dtype=torch.float16,
        device_map={'': torch.cuda.current_device()},  # 确保模型在正确的设备上
        load_in_8bit=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    # 直接使用output字典，不需要json.loads
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 得到训练集
train_df = pd.read_json("MedQA/dataset/medmcqa_train_new_3th.json", lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    #save_total_limit=2,  # 最多保存2个检查点
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    save_strategy="steps",  # 按步数保存
    evaluation_strategy="no",  # 不进行评估
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 训练前打印保存路径
print(f"模型将保存在: {output_dir}")

# 开始训练
trainer.train()

# 训练完成后保存最终模型
print("训练完成，正在保存最终模型...")
trainer.save_model(output_dir)

print(f"模型已保存到: {output_dir}")