import pandas as pd
import numpy as np
homework = pd.read_json('homework.json', lines=True)

# print(homework.head())

emotion_cols = homework.columns[homework.columns.str.contains('emotion')]
for col in emotion_cols:
	homework[col] = pd.to_numeric(homework[col], errors = 'coerce')
	homework[col] = np.where(homework[col] < 0, 0, homework[col])  # Replace negative values w 0.
	homework[col].fillna(0, inplace = True)  # Replace missing values w 0.


# print(homework.info())
# for col in emotion_cols:
# 	homework[col] = pd.to_numeric(homework[col], errors = 'coerce')
# 	homework[col] = np.where(homework[col] < 0, 0, homework[col])  # Replace negative values w 0.
# 	homework[col].fillna(0, inplace = True)  # Replace missing values w 0.

# 	print(homework[col].max())
# 	print(homework[col].min())
# print(homework[emotion_cols].info())
# print(homework[emotion_cols].sum(axis=1).mean())
# print(homework['emotion_9'].unique())
# print(homework['emotion_9'].value_counts())

# print('Since emotion 9 is nearly always 1, we should naivle predict 0,0,0...1. This would result in us missing \
# a bit over 1/10 each time. So a hamming loss of 10% is a lower bound on performance.') 

print(homework.worker_id.nunique())
print(homework.shape)