# coding: utf-8

import pandas as pd
from collections import Counter
import datetime
import numpy as np
get_ipython().magic('matplotlib inline')


from sklearn import preprocessing
from sklearn.metrics import log_loss



df = pd.read_csv('../Output/train1_step3.csv') # 数据加载
test_data = pd.read_csv('../Output/test1_step3.csv')

df.info()


# 将数据集按照context_timestamp的日期划分成训练集和测试集
# 18-23日的当成训练集，24日的当成测试集


df['context_time_day'] = [datetime.datetime.fromtimestamp(df['context_timestamp'][i]).day for i in range(df.shape[0])]



df1 = df.drop(['context_time_day', 'context_timestamp','instance_id'], 1)
test_data = test_data.drop(['context_timestamp'], 1)



df_label = df1.pop('is_trade')
df_data = df1
del df1



instance_df = test_data.pop('instance_id') # pd.series

rows = df_data.shape[0]
merge_data = df_data.append(test_data)



merge_data.info()



scale_data = preprocessing.scale(merge_data)

data = scale_data[0:rows]  # 训练集数据
test_data = scale_data[rows:] # 测试集数据



train_df = data[df['context_time_day']<=23]
test_df = data[df['context_time_day']>23]



train_df_label = df_label[df['context_time_day']<=23]
test_df_label = df_label[df['context_time_day']>23]



train_df



def generateResult(instance_df, test_prelabel, filename='../Output/LRpre_result_0321.txt'):
    result = pd.DataFrame({'instance_id':instance_df})
    result['predicted_score'] = test_prelabel
    result.to_csv(filename, index=False, sep=' ', line_terminator='\r')


# # GBDT
# Gradient Boosting Decision Tree梯度提升决策树
from sklearn import ensemble

# n_estimators树的颗数; learning_rate学习率; max_depth树的深度
GBDT = ensemble.GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=7)

GBDT.fit(train_df, train_df_label)

test_df_preLabel = GBDT.predict(test_df)
logLoss = log_loss(test_df_label, test_df_preLabel)
print (logLoss)
# 线下0.0823620123728 线上 0.08362

test_prelabel = GBDT.predict(test_data)

generateResult(instance_df, test_prelabel, filename='../Output/GBDTpre_result_0322.txt')


# # XGBoost
import xgboost as xgb


XGB_model = xgb.XGBRegressor(n_estimators=60, learning_rate=0.1, max_depth=7)

XGB_model.fit(train_df, train_df_label, eval_set=[(test_df, test_df_label)])

test_df_preLabel = XGB_model.predict(test_df)
logLoss = log_loss(test_df_label, test_df_preLabel)
print (logLoss)
# 线下 0.0841361944155  线上 0.0830
# 线下 0.0825883856602 线上 0.08324

test_prelabel = XGB_model.predict(test_data)
generateResult(instance_df, test_prelabel, filename='../Output/XGBpre_result_0325.txt')


# # lightGBM
import lightgbm as lgb

LGBM = lgb.LGBMRegressor(max_depth=7, n_estimators=60, learning_rate=0.1, num_leaves=63)

LGBM.fit(train_df, train_df_label)

test_df_preLabel = LGBM.predict(test_df)
logLoss = log_loss(test_df_label, test_df_preLabel)
print (logLoss)
# 线下0.08332 线上0.08337
# 线下0.0822370223983 线上0.08331

test_prelabel = LGBM.predict(test_data)
generateResult(instance_df, test_prelabel, filename='../Output/LGBMpre_result_0326.txt')


# # 随机森林RF
# Random Forest回归模型

from sklearn import ensemble

# n_estimators树的颗数; max_features寻找最佳划分的特征数; max_depth树的深度
RF = ensemble.RandomForestRegressor(n_estimators=30, max_features='log2', max_depth=7)

RF.fit(train_df, train_df_label)

test_df_preLabel = RF.predict(test_df)
logLoss = log_loss(test_df_label, test_df_preLabel)
print (logLoss)
# 线下 0.0828739759419 线下 0.08359

test_prelabel = RF.predict(test_data)
generateResult(instance_df, test_prelabel, filename='../Output/RFpre_result_0321.txt')

