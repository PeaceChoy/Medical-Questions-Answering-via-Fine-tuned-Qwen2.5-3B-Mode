import json
import pandas as pd

def load_jsonl(file_path):
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line)["Correct option"])
    return data

def calculate_accuracy(predictions, ground_truth):
    """计算准确率"""
    correct = 0
    total = 0
    
    for index, pred_answer in enumerate(predictions):
            total += 1
            # 获取真实答案
            gt_answer = ground_truth[index]
            if pred_answer == gt_answer:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def main():
    # 加载预测结果
    predictions = load_jsonl('MedQA/result/result_ftd_qwen.json')
    
    # 加载测试集
    test_data = load_jsonl('MedQA/dataset/medmcqa_train_test.json')
    
    # 计算准确率
    accuracy, correct, total = calculate_accuracy(predictions, test_data)
    
    # 打印结果
    print(f"总样本数: {total}")
    print(f"正确预测数: {correct}")
    print(f"准确率: {accuracy * 100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()

