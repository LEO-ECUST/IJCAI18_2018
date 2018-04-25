import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

import time


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def base_process(data):
    lbl = preprocessing.LabelEncoder()
    print(
        '--------------------------------------------------------------item--------------------------------------------------------------')
    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
    # for i in range(10):
    #   data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])
    print(
        '--------------------------------------------------------------user--------------------------------------------------------------')
    for col in ['user_id']:
        data[col] = lbl.fit_transform(data[col])

    print(
        '--------------------------------------------------------------context--------------------------------------------------------------')
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour
    data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))

    print(
        '--------------------------------------------------------------shop--------------------------------------------------------------')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    # data['shop_score_delivery0'] = data['shop_score_delivery'].apply(lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)
    return data


def map_hour(x):
    if (x >= 7) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 20):
        return 2
    else:
        return 3


def deliver(x):
    # x=round(x,6)
    jiange = 0.1
    for i in range(1, 20):
        if (x >= 4.1 + jiange * (i - 1)) & (x <= 4.1 + jiange * i):
            return i + 1
    if x == -5:
        return 1


def deliver1(x):
    if (x >= 2) & (x <= 4):
        return 1
    elif (x >= 5) & (x <= 7):
        return 2
    else:
        return 3


def review(x):
    # x=round(x,6)
    jiange = 0.02
    for i in range(1, 30):
        if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
            return i + 1
    if x == -1:
        return 1


def review1(x):
    # x=round(x,6)
    if (x >= 2) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 15):
        return 2
    else:
        return 3


def service(x):
    # x=round(x,6)
    jiange = 0.1
    for i in range(1, 20):
        if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
            return i + 1
    if x == -1:
        return 1


def service1(x):
    if (x >= 2) & (x <= 7):
        return 1
    elif (x >= 8) & (x <= 9):
        return 2
    else:
        return 3


def describe(x):
    # x=round(x,6)
    jiange = 0.1
    for i in range(1, 30):
        if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
            return i + 1
    if x == -1:
        return 1


def describe1(x):
    if (x >= 2) & (x <= 8):
        return 1
    elif (x >= 9) & (x <= 10):
        return 2
    else:
        return 3


def shijian(data):
    data['hour_map'] = data['hour'].apply(map_hour)
    return data


def shop_fenduan(data):
    data['shop_score_delivery'] = data['shop_score_delivery'] * 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply(deliver)
    data['deliver_map'] = data['deliver_map'].apply(deliver1)
    # del data['shop_score_delivery']
    print(data.deliver_map.value_counts())

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply(service)
    data['service_map'] = data['service_map'].apply(service1)
    # del data['shop_score_service']
    print(data.service_map.value_counts())  # 视为好评，中评，差评
    #
    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply(describe)
    data['de_map'] = data['de_map'].apply(describe1)
    # del data['shop_score_description']
    print(data.de_map.value_counts())

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply(review)
    data['review_map'] = data['review_map'].apply(review1)
    print(data.review_map.value_counts())

    data['normal_shop'] = data.apply(
        lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0,
        axis=1)
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']
    return data
def user_item(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')
    print('一个user有多少item_id,item_brand_id……Yes')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = (1 + data[str(col) + '_user_cnt']) / (100 + data['user_cnt'])
        del data[str(col) + '_user_cnt']

    print('一个user_gender有多少item_id,item_brand_id……NO')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = (1 + data[str(col) + '_user_gender_cnt']) / (
                    100 + data['user_gender_cnt'])
        del data[str(col) + '_user_gender_cnt']

    print('一个user_age_level有多少item_id,item_brand_id……NO')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = (1 + data[str(col) + '_user_age_cnt']) / (100 + data['user_age_cnt'])
        del data[str(col) + '_user_age_cnt']

    print('一个user_occupation_id有多少item_id,item_brand_id…NO')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = (1 + data[str(col) + '_user_occ_cnt']) / (100 + data['user_occ_cnt'])
        del data[str(col) + '_user_occ_cnt']

    return data


def user_shop(data):
    print('一个user有多少shop_id,shop_review_num_level……YES')

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = (1 + data[str(col) + '_user_cnt']) / (100 + data['user_cnt'])
        del data[str(col) + '_user_cnt']
    del data['user_cnt']

    print('一个user_gender有多少shop_id,shop_review_num_level……NO')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = (1 + data[str(col) + '_user_gender_cnt']) / (
                    100 + data['user_gender_cnt'])
        del data[str(col) + '_user_gender_cnt']
    del data['user_gender_cnt']

    print('一个user_age_level有多少shop_id,shop_review_num_level……NO')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = (1 + data[str(col) + '_user_age_cnt']) / (100 + data['user_age_cnt'])
        del data[str(col) + '_user_age_cnt']
    del data['user_age_cnt']

    print('一个user_occupation_id有多少shop_id,shop_review_num_level……NO')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = (1 + data[str(col) + '_user_occ_cnt']) / (100 + data['user_occ_cnt'])
        data[str(col) + '_user_occ_cnt']
    del data['user_occ_cnt']

    return data


def lgbCV1(train, test):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]

    X = train[col]
    y = train['is_trade'].values
    X_tes = test[col]
    y_tes = test['is_trade'].values
    print('Training LGBM model...')
    from sklearn.linear_model import LogisticRegression

    gbm = lgb.LGBMRegressor(objective='binary',
                            num_leaves=64,
                            learning_rate=0.01,
                            n_estimators=5000,
                            colsample_bytree=0.65,
                            subsample=0.75,
                            # reg_alpha = 0.4

                            )
    gbm.fit(X, y,
            eval_set=[(X_tes, y_tes)],
            eval_metric='binary_logloss',
            early_stopping_rounds=250)

    y_pred_1 = gbm.predict(X_tes, num_iteration=gbm.best_iteration_)
    print(log_loss(y_tes, y_pred_1))

    best_iter = gbm.best_iteration_

    return best_iter


def sub1(train, test, best_iter):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    X = train[col]
    y = train['is_trade'].values
    print('Training LGBM model...')
    gbm = lgb.LGBMRegressor(objective='binary',
                            num_leaves=64,
                            learning_rate=0.01,
                            n_estimators=best_iter,
                            colsample_bytree=0.65,
                            subsample=0.75,
                            # reg_alpha = 0.4

                            )
    gbm.fit(X, y)

    pred = gbm.predict(test[col])
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub = pd.read_csv("E:\\alimama\\round1_ijcai_18_test_b_20180418.txt", sep="\s+")
    sub = pd.merge(sub, sub1, on=['instance_id'], how='left')
    sub = sub.fillna(0)
    # sub[['instance_id', 'predicted_score']].to_csv('result/result0320.csv',index=None,sep=' ')
    sub[['instance_id', 'predicted_score']].to_csv('1233.txt', sep=" ", index=False)

def shijianhhh(data):
    print("----------------------------aaaaaaaaaaaaaaaa-----------------------------------")
    print("----------------------------aaaaaaaaaaaaaaaa-----------------------------------")
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'ds': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    item_shop_cnt = data.groupby(['day', 'user_id'], as_index=False)['instance_id'].agg(
        {'day12': 'count'})
    data = pd.merge(data, item_shop_cnt, on=['day', 'user_id'], how='left')
    data['day121'] = (1 + data['day12']) / (data['ds'] + 100)

    item_shop_cnt = data.groupby(['day', 'user_id', 'hour'], as_index=False)['instance_id'].agg(
        {'day13': 'count'})
    data = pd.merge(data, item_shop_cnt, on=['day', 'user_id', 'hour'], how='left')
    data['day131'] = (1 + data['day13']) / (data['day12'] + 100)
    del data['ds']
    return data

def iscon(x, y):
    t = -1
    for i in range(len(x)):
        if (x[i] == y):
            t = i
    return t


def h1(col):
    d1 = [-2, -2, -2, -2, -2, -2]
    s1 = col.item_category_list.split(';')
    for i in range(len(s1)):
        d1[2 * i] = -1
        d1[2 * i + 1] = -1
    s2 = col.item_property_list.split(';')
    s3 = col.predict_category_property.split(';')
    for i in range(len(s3)):
        s31 = s3[i].split(':')
        if (s31[0] != '-1'):
            s312 = s31[1].split(',')
            ifs = iscon(s1, s31[0])
            if (ifs >= 0):
                if (s312[0] == '-1'):
                    d1[2 * ifs] = 0
                    d1[2 * ifs + 1] = 0
                else:
                    co = 0
                    for j in range(len(s312)):
                        if (iscon(s2, s312[j]) >= 0):
                            co += 1
                    d1[2 * ifs] = len(s312)
                    d1[2 * ifs + 1] = co
    return d1


def coun(data):
    data['haha'] = data.apply(h1, axis=1)
    for i in range(0, 6):
        data['haha' + str(i)] = data['haha'].map(
            lambda x: x[i])  # item_category_list的第0列全部都一样
    del data['haha']
    # del data['haha4']
    # del data['haha5']
    data['hahah1'] = data.apply(lambda x: x.haha1 / x.haha0 if (x.haha0 > 0) else 0, axis=1)
    data['hahah2'] = data.apply(lambda x: x.haha3 / x.haha2 if (x.haha2 > 0) else 0, axis=1)
    return data

if __name__ == "__main__":
    train = pd.read_csv("E:\\alimama\\round1_ijcai_18_train_20180301.txt", sep="\s+")
    test = pd.read_csv("E:\\alimama\\round1_ijcai_18_test_b_20180418.txt", sep="\s+")
    data = pd.concat([train, test])
    data = data.drop_duplicates(subset='instance_id')  # 把instance id去重
    print('haha1121312')
    data = coun(data)
    data = base_process(data)
    data = shijian(data)
    data = shop_fenduan(data)
    data = shijianhhh(data)
    data = user_item(data)
    data = user_shop(data)
    print("----------------------------------------------------线下----------------------------------------")
    train = data[(data['day'] >= 18) & (data['day'] <= 23)]
    test = data[(data['day'] == 24)]

    best_iter = lgbCV1(train, test)
    "----------------------------------------------------线上----------------------------------------"
    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]
    sub1(train, test, best_iter)

