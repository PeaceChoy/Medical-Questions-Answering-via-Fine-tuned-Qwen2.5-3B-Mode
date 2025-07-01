import json

# 定义固定的instruction
INSTRUCTION = "You are a medical expert, please give professional answers based on the user's questions. The answer should include an analysis of the question and the correct option."

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        count = 0
        for line in infile:
            '''
            if count >= 5000:
                break
            '''
            if count >= 5500:
                break
            if count < 5000:
                count += 1
                continue
                
            # 读取每一行并解析JSON
            data = json.loads(line)
            
            input = f"""{data['question']}\nOptions:\nA) {data['opa']}\nB) {data['opb']}\nC) {data['opc']}\nD) {data['opd']}"""
            if data['cop'] < 0:
                correct_answer = ''
            else:
                correct_answer = ['A', 'B', 'C', 'D'][data['cop']]
            output = f"Analysis: {data['exp']}\nCorrect option: {correct_answer}"
            
            
            # 创建新的字典，包含instruction, input和output
            new_data = {
                "input": input,
                "Analysis": data['exp'],
                "Correct option": correct_answer
            }
            
            # 将新的字典写入输出文件
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')
            count += 1

# 使用示例
input_file = "MedQA/dataset/medmcqa_train.json"
output_file = "MedQA/dataset/medmcqa_train_test.json"

process_jsonl(input_file, output_file)
print(f"处理完成。输出文件：{output_file}")