# import json
# from rouge_score import rouge_scorer
# from bert_score import score
# import numpy as np
# from tqdm import tqdm
# import os

# def load_json(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         # 逐行读取JSON
#         for line in f:
#             line = line.strip()
#             if line:  # 确保行不为空
#                 try:
#                     item = json.loads(line)
#                     data.append(item)
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing line: {line[:100]}...")
#                     continue
#     return data

# def calculate_metrics(pred_text, ref_text):
#     # Calculate ROUGE scores
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     rouge_scores = scorer.score(pred_text, ref_text)
    
#     # Calculate BERTScore
#     P, R, F1 = score([pred_text], [ref_text], lang='en', verbose=False)
#     bert_score = {
#         'precision': P.item(),
#         'recall': R.item(),
#         'f1': F1.item()
#     }
    
#     return rouge_scores, bert_score

# def evaluate_analysis(pred_file, ref_file):
#     print(f"Loading prediction file: {pred_file}")
#     pred_data = load_json(pred_file)
#     print(f"Loading reference file: {ref_file}")
#     ref_data = load_json(ref_file)
    
#     print(f"Loaded {len(pred_data)} predictions and {len(ref_data)} references")
    
#     rouge1_scores = []
#     rouge2_scores = []
#     rougeL_scores = []
#     bert_f1_scores = []
    
#     for pred in tqdm(pred_data):
#         pred_id = pred.get('id')
#         ref = next((r for r in ref_data if r.get('id') == pred_id), None)
        
#         if ref and 'Analysis' in pred and 'Analysis' in ref:
#             pred_analysis = pred['Analysis']
#             ref_analysis = ref['Analysis']
            
#             if not pred_analysis or not ref_analysis:
#                 continue
                
#             try:
#                 rouge_scores, bert_score = calculate_metrics(pred_analysis, ref_analysis)
                
#                 rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
#                 rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
#                 rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
#                 bert_f1_scores.append(bert_score['f1'])
#             except Exception as e:
#                 print(f"Error processing id {pred_id}: {str(e)}")
#                 continue
    
#     if not rouge1_scores:
#         print("Warning: No valid scores were calculated!")
#         return {
#             'rouge1': 0.0,
#             'rouge2': 0.0,
#             'rougeL': 0.0,
#             'bert_score_f1': 0.0
#         }
    
#     metrics = {
#         'rouge1': np.mean(rouge1_scores),
#         'rouge2': np.mean(rouge2_scores),
#         'rougeL': np.mean(rougeL_scores),
#         'bert_score_f1': np.mean(bert_f1_scores)
#     }
    
#     return metrics

# def main():
#     dataset_dir = "dataset"
#     result_dir = "result"
    
#     ref_file = os.path.join(dataset_dir, "medmcqa_train_test.json")
#     pred_files = [
#         os.path.join(result_dir, "result_baseline.json"),
#         os.path.join(result_dir, "result_ftd_qwen.json"),
#         os.path.join(result_dir, "result_gpt.json")
#     ]
    
#     results = {}
#     for pred_file in pred_files:
#         print(f"\nEvaluating {pred_file}...")
#         metrics = evaluate_analysis(pred_file, ref_file)
#         results[pred_file] = metrics
    
#     print("\nEvaluation Results:")
#     print("=" * 50)
#     for pred_file, metrics in results.items():
#         print(f"\n{pred_file}:")
#         for metric_name, score in metrics.items():
#             print(f"{metric_name}: {score:.4f}")

# if __name__ == "__main__":
#     main()


import json
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
from tqdm import tqdm
import os

def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # 如果没有id，添加行号作为id
                    if 'id' not in item:
                        item['id'] = i
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:100]}...")
                    continue
    return data

def calculate_metrics(pred_text, ref_text):
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(pred_text, ref_text)
    
    # Calculate BERTScore
    P, R, F1 = score([pred_text], [ref_text], lang='en', verbose=False)
    bert_score = {
        'precision': P.item(),
        'recall': R.item(),
        'f1': F1.item()
    }
    
    return rouge_scores, bert_score

def evaluate_analysis(pred_files, ref_file):
    # 读取所有文件
    ref_data = load_json(ref_file)
    pred_data_dict = {file: load_json(file) for file in pred_files}
    
    # 获取qwen文件中Analysis为None的行号
    qwen_file = [f for f in pred_files if 'ftd_qwen' in f][0]
    none_analysis_indices = set()
    for item in pred_data_dict[qwen_file]:
        if item.get('Analysis') == 'None':
            none_analysis_indices.add(item['id'])
    
    print(f"Found {len(none_analysis_indices)} samples with None Analysis in qwen file")
    
    results = {}
    for pred_file in pred_files:
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bert_f1_scores = []
        
        for pred in tqdm(pred_data_dict[pred_file], desc=f"Processing {pred_file}"):
            pred_index = pred['id']
            
            # 如果行号在需要跳过的列表中，则跳过
            if pred_index in none_analysis_indices:
                continue
                
            if pred_index < len(ref_data):
                ref = ref_data[pred_index]
                
                pred_analysis = pred.get('Analysis', '')
                ref_analysis = ref.get('Analysis', '')
                
                if not pred_analysis or pred_analysis == 'None' or not ref_analysis:
                    continue
                    
                try:
                    rouge_scores, bert_score = calculate_metrics(pred_analysis, ref_analysis)
                    
                    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
                    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
                    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
                    bert_f1_scores.append(bert_score['f1'])
                except Exception as e:
                    print(f"Error processing index {pred_index}: {str(e)}")
                    continue
        
        if rouge1_scores:  # 只有在有分数时才计算平均值
            metrics = {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores),
                'bert_score_f1': np.mean(bert_f1_scores),
                'valid_samples': len(rouge1_scores)
            }
        else:
            metrics = {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bert_score_f1': 0.0,
                'valid_samples': 0
            }
        results[pred_file] = metrics
    
    return results

def main():
    dataset_dir = "dataset"
    result_dir = "result"
    
    ref_file = os.path.join(dataset_dir, "medmcqa_train_test.json")
    pred_files = [
        os.path.join(result_dir, "result_baseline.json"),
        os.path.join(result_dir, "result_ftd_qwen.json"),
        os.path.join(result_dir, "result_gpt.json")
    ]
    
    print("Evaluating files...")
    results = evaluate_analysis(pred_files, ref_file)
    
    print("\nEvaluation Results:")
    print("=" * 50)
    for pred_file, metrics in results.items():
        print(f"\n{pred_file}:")
        print(f"Number of valid samples: {metrics['valid_samples']}")
        for metric_name, score in metrics.items():
            if metric_name != 'valid_samples':
                print(f"{metric_name}: {score:.4f}")

if __name__ == "__main__":
    main()