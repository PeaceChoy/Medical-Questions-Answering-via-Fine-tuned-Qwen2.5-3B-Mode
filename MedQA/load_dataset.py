import pandas as pd
from datasets import load_dataset
# 加载数据集
ds = load_dataset("openlifescienceai/medmcqa")

# 将数据集转换为 Pandas DataFrame
df = pd.DataFrame(ds['train'])

# 保存为 JSON 文件
df.to_json("medmcqa_train.json", orient='records', lines=True)

# 如果需要保存其他分割（如验证集或测试集），可以重复上述步骤
df_val = pd.DataFrame(ds['validation'])
df_val.to_json("medmcqa_validation.json", orient='records', lines=True)

df_test = pd.DataFrame(ds['test'])
df_test.to_json("medmcqa_test.json", orient='records', lines=True)
