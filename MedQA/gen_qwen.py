from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from tqdm import tqdm
# 设置模型路径
base_model_path = "models/Qwen2.5-3B-Instruct"
lora_model_path = "MedQA/output/Qwen2.5-3B-3th"

# 加载基础模型和分词器
print("正在加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载LoRA权重
print("正在加载LoRA权重...")
model = PeftModel.from_pretrained(model, lora_model_path)

def generate_response(instruction, input_text):
    # 构建提示
    prompt = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response

# 测试示例
if __name__ == "__main__":
    # 示例问题
    #instruction = "You are a medical expert, please give professional answers based on the user's questions. The answer should include an analysis of the question and the correct option."
    instruction = """
    You are a medical expert, please give professional answers based on the user's questions. 
    The answer should include an analysis of the question and the correct option in the following format:

    Analysis: ...
    Correct option: ...
    """
    print("Start Generation:")
    count = 0
    with open("MedQA/dataset/medmcqa_train_test.json", 'r', encoding='utf-8') as infile, open("MedQA/result/result_ftd_qwen.json", 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=100):
            if count >= 100:
                break
            data = json.loads(line)
            question = data["input"]
            #print(question)
            response = generate_response(instruction, question)
            analysis_start = response.find('Analysis: ') + len('Analysis: ')
            analysis_end = response.find('Correct option: ')
            analysis = response[analysis_start:analysis_end].strip()
            correct_option_start = analysis_end + len('Correct option: ')
            correct_option = response[correct_option_start:correct_option_start+1].strip()
            response_dict = {"Analysis": analysis, "Correct option": correct_option}
            print(response)
            json.dump(response_dict, outfile, ensure_ascii=False)
            outfile.write('\n')

            count += 1
            
    
    '''print("问题:", question)
    response = generate_response(instruction, question)
    analysis_start = response.find('Analysis: ') + len('Analysis: ')
    analysis_end = response.find('Correct option: ')
    analysis = response[analysis_start:analysis_end].strip()
    correct_option_start = analysis_end + len('Correct option: ')
    correct_option = response[correct_option_start:].strip()
    response_dict = {"Analysis": analysis, "Correct option": correct_option}
    print("回答:\n", response) 
    print(response_dict)'''
    