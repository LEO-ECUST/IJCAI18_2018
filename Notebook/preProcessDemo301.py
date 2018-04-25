# coding: utf-8


import pandas as pd
from collections import Counter
import datetime
import numpy as np
get_ipython().magic('matplotlib inline')

df = pd.read_csv('../Output/train_step1.csv') # 训练集数据加载
test_df = pd.read_csv('../Data/round1_ijcai_18_test_a_20180301/round1_ijcai_18_test_a_20180301.txt', sep=' ') # 测试集数据加载


# 获得每个字段的缺失值样本数 原数据中缺失值为-1
def getDetailOfNAN(df):
    for col in df.columns:
        print (col + ': %d' % df[df[col]==-1].shape[0])
        
# 不同离散型特征具有不同的取值
def numberOfColValues(df):
    for col in df.columns:
        print (col + ': %d 特异值' % len(set(df[col])))

# 获得指定特征字段的详细情况，（特异值分布和样本分布）
def getDetailColumns(df, cols=['item_brand_id', 'item_city_id', 'item_id', 'user_id', 'context_id','shop_id','item_category_0','item_category_1','item_category_2','item_property_0','item_property_1','item_property_2','predict_category_property_0','predict_category_property_1','predict_category_property_2']):
    lens = df.shape[0]
    for col in cols:
        values = len(set(df[col]))
        print (col + '   %d   ' % values)
        print ("平均每个特异值有%d个样本数" % (lens/values))

# # 对dataframe数据进行统计描述：
df.describe()

#  某些字段的取值范围可以进行缩放
#     user_age_level-1000
#     user_occupation_id-2000
#     user_star_level-3000
#     context_page_id-4000
#     shop_star_level-4999

# 获得每个字段的缺失值样本数 原数据中缺失值为-1
getDetailOfNAN(df)

# 具有缺失值的特征列为：
#     item_city_id,item_brand_id,item_sales_level,
#     user_gender_id,user_age_level,user_occupation_id,user_star_level,
#     shop_review_positive_rate,shop_score_service,shop_score_delivery,shop_score_description

# 缺失值处理方法：
#     1、直接删除，不超过样本数的2%,若缺失值占95%且特征不是很重要，则可以选择删除整条特征列
#     2、采用均值或者中位数（统计值）来进行填补
#     3、忽略不作处理
#     4、就近选取最相近的对象最近填补或者K近邻均值填补
#     5、建模预测：回归、随机森林、聚类均值等

# # 缺失值处理
# 缺失值的样本量占总样本量的百分比很小，可以直接删除
#     shop_score_service: 59
#     shop_score_delivery: 59
#     shop_score_description: 59
#     shop_review_positive_rate: 7
#     item_brand_id: 473
#     item_city_id: 277

# 对训练集中部分特征值为NAN的样本进行删除
df = df[df['shop_review_positive_rate']!=-1]
df = df[df['shop_score_delivery']!=-1]
df = df[df['shop_score_description']!=-1]
df = df[df['shop_score_service']!=-1]
df = df[df['item_brand_id']!=-1]
df = df[df['item_city_id']!=-1]

df = df.reset_index(drop=True) # 删除部分具有缺失值样本后重新更新数据集的index


# # 3 对部分字段进行数据标准化等转化方式
# 归一化

def dataNormalization(df):
    df.user_age_level = df.user_age_level.replace(-1,999)
    df.user_age_level = df.user_age_level - 1000

    df.user_occupation_id = df.user_occupation_id.replace(-1,1999)
    df.user_occupation_id = df.user_occupation_id - 2000

    df.user_star_level = df.user_star_level.replace(-1,2999)
    df.user_star_level = df.user_star_level - 3000

    df.context_page_id = df.context_page_id - 4000

    df.shop_star_level = df.shop_star_level - 4999
    
    return df

df = dataNormalization(df)
test_df = dataNormalization(test_df)

# 缺失值的样本量较多
# 
#     item_sales_level
#     user_gender_id
#     user_age_level
#     user_occupation_id
#     user_star_level
# 决策树系列对缺失值的敏感度不是很强，不处理忽略

# # 特征提取
# 数据转化
# 
#     item_category_list
#     item_property_list
#     predict_category_property
# 是广告类目列表，根目录;子目录;子子目录
# 对该字段进行分割，提取叶节点

def dataConvert(df):
    for i in range(3):
        df['item_category_%d'%i] = df['item_category_list'].apply(lambda str1: int(str1.split(';')[i]) if len(str1.split(';'))>i else -1)
    df = df.drop(['item_category_list'], 1)
    for i in range(3):
        df['item_property_%d'%i] = df['item_property_list'].apply(lambda str1: int(str1.split(';')[i]) if len(str1.split(';'))>i else -1)
    df = df.drop(['item_property_list'], 1)
    for i in range(3):
        df['predict_category_property_%d'%i] = df['predict_category_property'].apply(lambda str1: int(str1.split(';')[i].split(':')[0])                                                                                      if len(str1.split(';'))>i else -1)
    df = df.drop(['predict_category_property'], 1)
    return df

df = dataConvert(df)
test_df = dataConvert(test_df)


# 不同离散型特征具有不同的取值
# numberOfColValues(df)
numberOfColValues(test_df)


# 数据类型转化
#     context_timestamp
# 将context_timestamp字段进行维度转化成hour/min/hour*60+min和day


def timeStampConvert(df):
    df['context_time'] = [datetime.datetime.fromtimestamp(df['context_timestamp'][i]).hour * 60 +                                datetime.datetime.fromtimestamp(df['context_timestamp'][i]).minute for i in range(df.shape[0])]
    df['context_time_day'] = [datetime.datetime.fromtimestamp(df['context_timestamp'][i]).day for i in range(df.shape[0])]
    df['context_time_weekday'] = [datetime.datetime.fromtimestamp(df['context_timestamp'][i]).weekday() for i in range(df.shape[0])]
    return df

df = timeStampConvert(df)
test_df = timeStampConvert(test_df)

# 特殊变量（字符转化变量）-不连续数字或者文本
#     item_brand_id
#     item_city_id
#     
#     item_id
#     user_id
#     context_id 舍弃
#     shop_id
#     
#     item_category_0: 1 特异值    舍弃
#     item_category_1: 13 特异值
#     item_category_2: 3 特异值
#     item_property_0: 216 特异值
#     item_property_1: 122 特异值
#     item_property_2: 189 特异值
#     predict_category_property_0: 283 特异值
#     predict_category_property_1: 396 特异值
#     predict_category_property_2: 457 特异值


rows = df.shape[0]
label = df['is_trade']
df1 = df.drop(['is_trade'], 1).append(test_df)

# 获得指定特征字段的详细情况，（特异值分布和样本分布）
getDetailColumns(df1)

# context_id基本为样例的标识值，对应文本的编号，1个样本平均对应1个context_id，故舍弃
# item_category_0为样本的特定值，变动不大，训练集中几乎都一样

def dropCols(df):
    return df.drop(['context_id','item_category_0'], 1)

df1 = dropCols(df1)


# 对不连续的数字或者文本进行标号LabelEncoder
#     item_id
#     item_city_id
#     item_brand_id
#     item_category_1, item_category_2
#     item_property_0, item_property_1, item_property_2
#     user_id
#     shop_id
#     predict_category_property_0, predict_category_property_1, predict_category_property_2
# 

def dataLabelEncoder(df):
    df['item_category_1'] = pd.Categorical(df['item_category_1']).codes
    df['item_category_2'] = pd.Categorical(df['item_category_2']).codes

    df['item_property_0'] = pd.Categorical(df['item_property_0']).codes
    df['item_property_1'] = pd.Categorical(df['item_property_1']).codes
    df['item_property_2'] = pd.Categorical(df['item_property_2']).codes

    df['predict_category_property_0'] = pd.Categorical(df['predict_category_property_0']).codes
    df['predict_category_property_1'] = pd.Categorical(df['predict_category_property_1']).codes
    df['predict_category_property_2'] = pd.Categorical(df['predict_category_property_2']).codes

    df['shop_id'] = pd.Categorical(df['shop_id']).codes
    df['item_brand_id'] = pd.Categorical(df['item_brand_id']).codes
    df['item_city_id'] = pd.Categorical(df['item_city_id']).codes
    df['user_id'] = pd.Categorical(df['user_id']).codes
    df['item_id'] = pd.Categorical(df['item_id']).codes
    return df
df_enc = dataLabelEncoder(df1)
del df1


# 离散变量：有序特征--分类数据
#     user_occupation_id
#     user_gender_id
#   连续特征
#   
#     item_price_level
#     item_sales_level (-1)
#     item_collected_level
#     item_pv_level
#     user_age_level (-1)
#     user_star_level (-1)
#     shop_review_num_level
#     shop_star_level
#    
#     context_page_id
#     
#     context_time_weekday
#     context_time


# 不同离散型特征具有不同的取值
numberOfColValues(df_enc)


label = pd.DataFrame({'is_trade':label})
test_df = df_enc[rows:]
df = df_enc[0:rows].merge(label, left_index=True, right_index=True)
del df_enc, rows



# 连续变量：
# 
#     shop_review_positive_rate
#     shop_score_service
#     shop_score_delivery
#     shop_score_description
#     


# 对连续变量特征进行直方图分析
import matplotlib.pyplot as plt 

plt.figure(figsize=(12,18))
plt.subplot(4,1,1)
plt.hist(df['shop_review_positive_rate'], 100)
plt.subplot(4,1,2)
plt.hist(df['shop_score_service'], 100)
plt.subplot(4,1,3)
plt.hist(df['shop_score_delivery'], 100)
plt.subplot(4,1,4)
plt.hist(df['shop_score_description'], 100)
plt.show()


# 根据连续变量的直方图分析，对连续变量进行二值化处理，针对LR模型

df[['shop_review_positive_rate','shop_score_service', 'shop_score_delivery', 'shop_score_description']].describe()



df = df[df.shop_review_positive_rate>=0.95]
df = df[df.shop_score_service>=0.925]
df = df[df.shop_score_delivery>=0.9]
df = df[df.shop_score_description>=0.91]
# df['shop_review_positive_rate'] = df['shop_review_positive_rate'].apply(lambda x:int(x>=0.95)
# df['shop_score_service'] = df['shop_score_service'].apply(lambda x:int(x>=0.925)
# df['shop_score_delivery'] = df['shop_score_delivery'].apply(lambda x:int(x>=0.9)
# df['shop_score_description'] = df['shop_score_description'].apply(lambda x:int(x>=0.91)


# # 数据处理完毕，保存
df.to_csv('../Output/train1_step3.csv', index=False)
test_df.to_csv('../Output/test1_step3.csv', index=False)

