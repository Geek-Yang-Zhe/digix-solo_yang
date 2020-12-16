import pandas as pd
from get_feature_names import get_sparse_feature_names, get_pair_click_feature_names, get_pair_time_feature_names, \
    get_id_feature_names, get_click_feature_names, get_dense_feature_names, get_user_base_feature_names, \
    get_user_profile_feautre_names, \
    get_adv_base_feature_names, get_background_feature_names
import time
import numpy as np
from reduce_mem import reduce_mem
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
import datetime

sparse_feature_names = get_sparse_feature_names()
click_feature_names = get_click_feature_names()
pair_click_feature_names = get_pair_click_feature_names()
id_feature_names = get_id_feature_names()
dense_feature_names = get_dense_feature_names()

all_prefix = './yangzhe/feature/all/'


def walson_ctr(num_click, num_pv, z=1.96):
    p = num_click * 1.0 / num_pv
    n = num_pv
    A = p + z ** 2 / (2 * n)
    B = np.sqrt(p * (1 - p) / n + z ** 2 / (4 * (n ** 2)))
    C = z * B
    D = 1 + z ** 2 / n
    ctr = (A - C) / D
    return ctr


def get_max_appear_times(labels):
    max_time = 0
    max_his = 0
    for label in labels:
        if label == 0:
            max_time += 1
        else:
            if max_his < max_time:
                max_his = max_time
            max_time = 0
    return max_his if max_his > max_time else max_time


# 连续没点击的次数列表
def get_no_click_list(labels):
    max_time = 0
    ls = []
    for label in labels:
        if label == 0:
            max_time += 1
        elif max_time != 0:
            ls.append(max_time)
            max_time = 0
    if max_time != 0:
        ls.append(max_time)
    if len(ls) == 0:
        ls.append(0)
    ls1 = [np.max(ls), np.mean(ls), np.var(ls)]
    return ls1


class GenerateFeatures(object):
    def __init__(self):
        self.feature_path_prefix = ''
        self.generate_feature_from_past_days(day=2)
        self.generate_feature_from_past_days(day=3)
        self.generate_feature_from_past_days(day=4)
        self.generate_feature_from_past_days(day=5)
        self.generate_feature_from_past_days(day=6)
        self.generate_feature_from_past_days(day=7)
        self.generate_feature_from_past_days(day=8)

    def generate_feature_from_past_days(self, day):
        if day < 8:
            self.feature_path_prefix = './yangzhe/feature/{}th_feature/'.format(day)
        else:
            self.feature_path_prefix = './yangzhe/feature/8_9th_feature/'
        # 这是固定区间提取的特征，B榜到了也无需重新跑训练集中的这些,这些特征从day这一天前面的来提取pastday和lastday特征
        # 从day这一天提取当天曝光特征
        # 由于B榜是第九天的，对于all_feature来说没有第八天的标签，因此把他当成第8天来处理
        train_sparse_df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        if day < 8:
            train = train_sparse_df[train_sparse_df['pt_d'] == day]
            data = train_sparse_df[train_sparse_df['pt_d'] < day]
            last_day_df = train_sparse_df[train_sparse_df.pt_d == day - 1]
        else:
            train = train_sparse_df[train_sparse_df['pt_d'] >= 8]
            data = train_sparse_df[train_sparse_df['pt_d'] < 8]
            last_day_df = train_sparse_df[train_sparse_df.pt_d == 7]

        # part1部分特征
        self.generate_statistic_ctr_feature(train=train, data=data, day=day)
        self.generate_ctr_feature(train=train, data=data, day=day)
        self.generate_pair_ctr_feature(train=train, data=data, day=day, suffix='pastday')
        self.generate_continue_no_click_feature(train=train, data=data, day=day)
        self.generate_pair_continue_no_click_feature(train=train, data=data, day=day)
        self.generate_clk_time_interval_feature(train=train, data=data, day=day)
        self.generate_pair_clk_time_interval_feature(train=train, data=data, day=day)
        self.generate_ctr_cnt_rank_feature(train=train, data=data, day=day)

        # self.generate_statistics_cnt_feature(train=train, data=data, day=day)
        # self.generate_cnt_feature(train=train, data=data, day=day)

        # # 昨天的特征
        self.generate_lastday_ctr_feature(train=train, data=last_day_df, day=day)
        self.generate_pair_lastday_feature(train=train, data=last_day_df, day=day, suffix='lastday', cnt=True, clk=True)
        self.generate_lastday_ctr_cnt_rank_feature(train=train, data=last_day_df, day=day)

        # 今天的特征
        today_df = train.copy()
        self.generate_today_feature(train=train, data=today_df, day=day)
        self.generate_pair_today_feature(train=train, data=today_df, day=day)
        # self.generate_today_cnt_rank_feature(train=train, data=today_df, day=day)
        # self.generate_today_statistics_cnt_feature(train=train, data=today_df, day=day)
        print('第{}天处理完成'.format(day))

    def generate_w2v_feature(self, day, train_index):
        all_w2v_feature = pd.read_pickle(all_prefix + 'all_w2v_embedding_feature.pkl')
        df = reduce_mem(all_w2v_feature.loc[train_index].reset_index(drop=True))
        df.to_pickle(self.feature_path_prefix + '{}th_w2v_embedding.pkl'.format(day))

    def generate_n2v_feature(self, day, train_index):
        all_n2v_feature = pd.read_pickle(all_prefix + 'all_LINE_clk_embedding.pkl')
        df = reduce_mem(all_n2v_feature.loc[train_index].reset_index(drop=True))
        df.to_pickle(self.feature_path_prefix + '{}th_LINE_clk_embedding.pkl'.format(day))

    def generate_hist_behavior(self, train, data, day):
        # 获取用户关于场景和广告的历史点击行为，应该将历史信息按时间先倒序排好
        data = data.sort_values('pt_d', ascending=False, kind='mergesort')
        behavior_feature_name = get_adv_base_feature_names() + get_background_feature_names()
        hist_feature_name = ['hist_' + x for x in behavior_feature_name]

        click_df = data.loc[data['label'] == 1, ['uid'] + behavior_feature_name].rename(
            columns=dict(zip(behavior_feature_name, hist_feature_name)))
        click_df[hist_feature_name] = click_df[hist_feature_name].astype(str)
        click_hist = click_df.groupby('uid', as_index=False, sort=False).agg('^'.join)
        print('groupby完成:', datetime.datetime.now())

        behavior_df = pd.merge(train[['uid']], click_hist, on='uid', how='left')[hist_feature_name]
        behavior_df.fillna('0', inplace=True)

        all_hist_feature = []
        for feature_name in hist_feature_name:
            varlen_list = [[int(x) for x in y.split('^')] for y in behavior_df[feature_name]]
            varlen_list = pad_sequences(varlen_list, maxlen=5, padding='post', truncating='post')
            print(feature_name, datetime.datetime.now())
            all_hist_feature.append(varlen_list)

        with open(self.feature_path_prefix + '{}th_hist_behavior.pkl'.format(day), 'wb') as f:
            pickle.dump(all_hist_feature, f)

    def generate_cross_cnt_feature(self, day):
        feature_df = pd.DataFrame()
        data = pd.read_pickle(self.feature_path_prefix + '{}th_cnt.pkl'.format(day))
        # 交叉特征在单特征计数下的的占比，例如count(uid,task_id)/count(uid)，count(uid,task_id)/count(task_id)
        for i in ['uid']:
            single_name_0 = "{}_cnt".format(i)
            for j in get_adv_base_feature_names():
                single_name_1 = "{}_cnt".format(j)
                feature_pair = "_".join([i, j])
                col_name = "cnt-2order_{}".format(feature_pair)
                feature_df['cross_cnt_{}_0'.format(feature_pair)] = data[col_name] / data[single_name_0]
                feature_df['cross_cnt_{}_1'.format(feature_pair)] = data[col_name] / data[single_name_1]
        for i in ['uid'] + get_user_base_feature_names() + get_user_profile_feautre_names():
            single_name_0 = "{}_cnt".format(i)
            for j in get_background_feature_names():
                single_name_1 = "{}_cnt".format(j)
                feature_pair = "_".join([i, j])
                col_name = "cnt-2order_{}".format(feature_pair)
                feature_df['cross_cnt_{}_0'.format(feature_pair)] = data[col_name] / data[single_name_0]
                feature_df['cross_cnt_{}_1'.format(feature_pair)] = data[col_name] / data[single_name_1]
        for i in get_background_feature_names():
            single_name_0 = "{}_cnt".format(i)
            for j in get_adv_base_feature_names():
                single_name_1 = "{}_cnt".format(j)
                feature_pair = "_".join([i, j])
                col_name = "cnt-2order_{}".format(feature_pair)
                feature_df['cross_cnt_{}_0'.format(feature_pair)] = data[col_name] / data[single_name_0]
                feature_df['cross_cnt_{}_1'.format(feature_pair)] = data[col_name] / data[single_name_1]

        reduce_mem(feature_df).to_pickle(self.feature_path_prefix + '{}th_cross_cnt.pkl'.format(day))

    def generate_cnt_feature(self, train, data, day):
        feature_df = pd.DataFrame()
        # 交叉计数
        for i in ['uid']:
            for j in get_adv_base_feature_names():
                feature_pair = [i, j]
                col_name = "cnt-2order_{}".format("_".join(feature_pair))
                tmp_df = data.groupby(feature_pair, as_index=False, sort=False)['label'].agg({col_name: 'count'})
                feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair, how='left')[col_name]
        for i in ['uid']:
            for j in get_background_feature_names():
                feature_pair = [i, j]
                col_name = "cnt-2order_{}".format("_".join(feature_pair))
                tmp_df = data.groupby(feature_pair, as_index=False, sort=False)['label'].agg({col_name: 'count'})
                feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair, how='left')[col_name]

        reduce_mem(feature_df).to_pickle(self.feature_path_prefix + '{}th_cnt.pkl'.format(day))

    def generate_statistics_cnt_feature(self, train, data, day):
        # 这个特征nunique_0缺失较多
        uid_statistics_ls = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                             'slot_id', 'spread_app_id', 'tags', 'app_second_class']
        feature_df = pd.DataFrame()
        for feature in uid_statistics_ls:
            feature_pair = ['uid', feature]
            col_name = "cnt_{}_nunique_0".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[0], as_index=True, sort=False)[feature_pair[1]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[0], how='left')[col_name]
            col_name = "cnt_{}_nunique_1".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[1], as_index=True, sort=False)[feature_pair[0]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[1], how='left')[col_name]

        task_statistics_ls = ['slot_id', 'device_name', 'device_size', 'net_type',
                              'residence', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                              'communication_avgonline_30d', 'indu_name', 'age', 'city', 'career', 'gender']
        for feature in task_statistics_ls:
            feature_pair = ['task_id', feature]
            col_name = "cnt_{}_nunique_0".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[0], as_index=True, sort=False)[feature_pair[1]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[0], how='left')[col_name]
            col_name = "cnt_{}_nunique_1".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[1], as_index=True, sort=False)[feature_pair[0]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[1], how='left')[col_name]

        reduce_mem(feature_df).to_pickle(self.feature_path_prefix + '{}th_statistics_cnt.pkl'.format(day))

    def generate_statistic_ctr_feature(self, train, data, day):
        # 每个用户点击/曝光不同广告,素材等广告特征的数量，点击率，反映了用户的兴趣广泛程度
        uid_statistics_ls = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                             'slot_id', 'spread_app_id', 'tags', 'app_second_class']
        uid_statistics_df = self.generate_statistics_part_feature(data=data, key=['uid'],
                                                                  feature_names=uid_statistics_ls)
        # 每个广告任务被不同用户/年龄/城市 点击/曝光的数量，反映了广告的热度和普适度，这部分缺失有点严重
        task_statistics_ls = ['task_id', 'uid', 'slot_id', 'device_name', 'device_size', 'net_type',
                              'residence', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                              'communication_avgonline_30d']
        task_statistics_df = self.generate_statistics_part_feature(data=data, key=['task_id'],
                                                                   feature_names=task_statistics_ls)
        df1 = pd.merge(train['uid'], uid_statistics_df, on='uid', how='left').drop(columns=['uid'])
        df2 = pd.merge(train['task_id'], task_statistics_df, on='task_id', how='left').drop(columns=['task_id'])
        df = pd.concat([df1, df2], axis=1, sort=False)

        df = reduce_mem(df)
        df.to_pickle(self.feature_path_prefix + '{}th_statistics_ctr.pkl'.format(day))

    def generate_statistics_part_feature(self, data, key, feature_names):
        statistics_feature = [x for x in feature_names if x not in key]
        cnt_name = key + [feature_name + '_{}_statistics_cnt'.format('_'.join(key)) for feature_name in
                          statistics_feature]
        clk_name = key + [feature_name + '_{}_statistics_clk'.format('_'.join(key)) for feature_name in
                          statistics_feature]
        cnt_data = data.loc[:, feature_names].copy()
        clk_data = data.loc[data.label == 1, feature_names].copy()
        cnt_df = self.generate_unique_feature(cnt_data, key=key).rename(
            columns=dict(zip(feature_names, cnt_name)))
        clk_df = self.generate_unique_feature(clk_data, key=key).rename(
            columns=dict(zip(feature_names, clk_name)))
        statistics_df = pd.merge(cnt_df, clk_df, on=key, how='left')
        statistics_df.fillna(0, inplace=True)
        for i, feature_name in enumerate(statistics_feature):
            statistics_df[feature_name + '_{}_statistics_ctr'.format('_'.join(key))] = walson_ctr(
                statistics_df[feature_name + '_{}_statistics_clk'.format('_'.join(key))],
                statistics_df[feature_name + '_{}_statistics_cnt'.format('_'.join(key))])
            del statistics_df[feature_name + '_{}_statistics_cnt'.format('_'.join(key))]
        return statistics_df

    def generate_unique_feature(self, data, key):
        data = data.groupby(key, as_index=True, sort=False).nunique()
        data.drop(columns=key, inplace=True)
        data.reset_index(drop=False, inplace=True)
        return data

    def generate_ctr_feature(self, train, data, day):
        dense_names = []
        single_click_feature = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                                'slot_id', 'spread_app_id', 'age', 'city', 'device_name', 'device_size', 'career',
                                'residence', 'his_app_size', 'emui_dev', 'list_time',
                                'up_life_duration', 'consume_purchase', 'communication_avgonline_30d', 'indu_name']
        for feature_name in single_click_feature:
            # 这里保留单特征的平滑点击率
            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg({'clk': 'sum', 'cnt': 'count'})
            tmp_df['ctr'] = walson_ctr(tmp_df['clk'].to_numpy(), tmp_df['cnt'].to_numpy()).astype('float32')
            train = pd.merge(train, tmp_df, on=feature_name, how='left')
            train = train.rename(columns={'clk': feature_name + '_clk', 'ctr': feature_name + '_ctr',
                                          'cnt': feature_name + '_cnt'})
            dense_names += [feature_name + '_clk', feature_name + '_ctr']

        df = reduce_mem(train[dense_names])
        df.to_pickle(self.feature_path_prefix + '{}th_single_ctr.pkl'.format(day))

    def generate_pair_ctr_feature(self, train, data, day, suffix, cnt=False, clk=False):
        user_adv_df = self.pair_ctr_feature(train, data, fea_1=['uid'],
                                            fea_2=get_adv_base_feature_names(),
                                            suffix=suffix, cnt=cnt, clk=clk)

        user_background_df = self.pair_ctr_feature(train, data, fea_1=['uid'] + get_user_base_feature_names(),
                                                   fea_2=get_background_feature_names(),
                                                   suffix=suffix, cnt=cnt, clk=clk)

        background_adv_df = self.pair_ctr_feature(train, data, fea_1=get_background_feature_names(),
                                                  fea_2=get_adv_base_feature_names(),
                                                  suffix=suffix, cnt=cnt, clk=clk)
        reduce_mem(pd.concat([user_adv_df, user_background_df, background_adv_df], axis=1)). \
            to_pickle(self.feature_path_prefix + '{}th_pair_ctr_{}.pkl'.format(day, suffix))

    def pair_ctr_feature(self, train, data, fea_1, fea_2, cnt=False, clk=False, suffix=''):
        # 这里将分成三组,id特征到id特征,用户画像到用户画像,id特征到用户画像
        # 先提取id特征的ctr和cnt看下
        dense_names = []
        for i in range(len(fea_1)):
            for j in range(len(fea_2)):
                feature_pair = [fea_1[i], fea_2[j]]
                feature_pair_ctr = '_'.join(feature_pair + ['ctr', suffix])
                feature_pair_clk = '_'.join(feature_pair + ['clk', suffix])
                feature_pair_cnt = '_'.join(feature_pair + ['cnt', suffix])
                tmp_df = data.groupby(feature_pair, as_index=False, sort=False)['label']. \
                    agg({feature_pair_clk: 'sum', feature_pair_cnt: 'count'})
                tmp_df[feature_pair_ctr] = walson_ctr(tmp_df[feature_pair_clk].to_numpy(),
                                                      tmp_df[feature_pair_cnt].to_numpy()) \
                    .astype('float32')
                if cnt is True:
                    dense_names += [feature_pair_cnt]
                if clk is True:
                    dense_names += [feature_pair_clk]
                train = pd.merge(train, tmp_df, on=feature_pair, how='left')
                dense_names += [feature_pair_ctr]

        return train[dense_names]

    def generate_today_feature(self, train, data, day):
        # 当天曝光的次数，对于测试集没法提取当天点击，所以提取当天曝光次数，这个可以取特征组合(用户+广告id类)
        # 考虑其他特征/特征组合的当天曝光意义
        # 感觉这个特征可能过拟合严重，七号和八号的曝光次数分布可能很不一致，但实验结果还不错
        dense_names = []
        today_feature_names = ['uid'] + get_user_base_feature_names() + get_adv_base_feature_names()
        today_feature_names.remove('app_first_class')
        today_feature_names.remove('app_score')
        # 单特征
        for feature_name in today_feature_names:
            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg(
                {feature_name + '_today_cnt': 'count'})
            train = pd.merge(train, tmp_df, on=feature_name, how='left')
            dense_names += [feature_name + '_today_cnt']

        reduce_mem(train[dense_names]).to_pickle(self.feature_path_prefix + '{}th_today_cnt.pkl'.format(day))

    def generate_pair_today_feature(self, train, data, day):
        user_adv_df = self.pair_today_feature(train, data, fea_1=['uid'],
                                              fea_2=get_adv_base_feature_names())
        user_background_df = self.pair_today_feature(train, data, fea_1=['uid'] + get_user_base_feature_names(),
                                                     fea_2=get_background_feature_names())
        background_adv_df = self.pair_today_feature(train, data, fea_1=get_background_feature_names(),
                                                    fea_2=get_adv_base_feature_names())

        df = pd.concat([user_adv_df, user_background_df, background_adv_df], axis=1)
        reduce_mem(df).to_pickle(self.feature_path_prefix + '{}th_pair_today_cnt.pkl'.format(day))

    def pair_today_feature(self, train, data, fea_1, fea_2):
        dense_names = []
        for i in fea_1:
            for j in fea_2:
                feature_pair = [i, j]
                feature_pair_cnt = '_'.join(feature_pair) + '_today_cnt'
                tmp_df = data.groupby(feature_pair, as_index=False, sort=False)['label']. \
                    agg({feature_pair_cnt: 'count'})
                train = pd.merge(train, tmp_df, on=feature_pair, how='left')

                dense_names += [feature_pair_cnt]
        return train[dense_names]

    def generate_lastday_ctr_feature(self, train, data, day):
        # 昨天曝光的次数.点击次数,点击率
        dense_names = []
        click_feature = ['uid', 'task_id', 'adv_id', 'adv_prim_id', 'dev_id',
                         'spread_app_id', 'age', 'city', 'device_name', 'device_size', 'career',
                         'residence', 'his_app_size', 'list_time', 'up_life_duration']
        for feature_name in click_feature:
            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg({'clk': 'sum', 'cnt': 'count'})
            tmp_df['ctr'] = walson_ctr(tmp_df['clk'].to_numpy(), tmp_df['cnt'].to_numpy()).astype('float32')
            train = pd.merge(train, tmp_df, on=feature_name, how='left')
            train = train.rename(columns={'clk': feature_name + '_lastday_clk', 'ctr': feature_name + '_lastday_ctr',
                                          'cnt': feature_name + '_lastday_cnt'})
            dense_names += [feature_name + '_lastday_clk', feature_name + '_lastday_ctr', feature_name + '_lastday_cnt']
        reduce_mem(train[dense_names]).to_pickle(self.feature_path_prefix + '{}th_lastday_clk.pkl'.format(day))

    def generate_pair_lastday_feature(self, train, data, day, suffix, cnt=False, clk=False):
        adv_feature = ['task_id', 'adv_id', 'adv_prim_id', 'dev_id', 'spread_app_id', 'tags', 'indu_name']
        user_feature = ['age', 'city', 'device_name', 'residence', 'emui_dev', 'list_time', 'device_price']

        user_background_df = self.pair_ctr_feature(train, data, fea_1=['uid'] + user_feature,
                                                   fea_2=['slot_id'],
                                                   suffix=suffix, cnt=cnt, clk=clk)

        background_adv_df = self.pair_ctr_feature(train, data, fea_1=['slot_id'],
                                                  fea_2=adv_feature,
                                                  suffix=suffix, cnt=cnt, clk=clk)
        reduce_mem(pd.concat([user_background_df, background_adv_df], axis=1)). \
            to_pickle(self.feature_path_prefix + '{}th_pair_lastday_clk.pkl'.format(day))

    def generate_continue_no_click_feature(self, train, data, day):
        # 连续曝光未点击的最大次数，均值，方差，可以交叉，反映了用户的活跃度，广告的受欢迎度
        # 记得排序
        dense_names = []
        continue_no_click_ls = ['uid', 'task_id', 'adv_id', 'adv_prim_id', 'dev_id',
                                'slot_id', 'spread_app_id', 'age', 'city', 'device_name', 'device_size', 'career',
                                'residence', 'emui_dev', 'up_life_duration',
                                'consume_purchase', 'communication_avgonline_30d']
        for feature_name in continue_no_click_ls:
            tmp_feature = '{}_continue_no_click_list'.format(feature_name)
            no_click_feature_names = [feature_name + '_continue_no_click_' + x for x in
                                      ['max', 'mean', 'var']]
            dense_names += no_click_feature_names

            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg(
                {tmp_feature: get_no_click_list})
            no_click_feature_df = pd.DataFrame(tmp_df[tmp_feature].tolist(), columns=no_click_feature_names)
            no_click_feature_df[feature_name] = tmp_df[feature_name]
            train = pd.merge(train, no_click_feature_df, on=feature_name, how='left')
        reduce_mem(train[dense_names]).to_pickle(self.feature_path_prefix + '{}th_continue_no_clk.pkl'.format(day))

    def generate_pair_continue_no_click_feature(self, train, data, day):
        user_feature = ['age', 'city', 'device_name', 'device_size', 'career',
                        'residence', 'emui_dev', 'up_life_duration', 'consume_purchase', 'communication_avgonline_30d']
        adv_feature = ['task_id', 'adv_id', 'adv_prim_id', 'dev_id', 'spread_app_id']

        # 与slot_id结合的特征表现都很好
        adv_background_df = self.pair_continue_no_click(train, data, fea_1=adv_feature, fea_2=['slot_id'])

        user_background_df = self.pair_continue_no_click(train, data, fea_1=user_feature + ['uid'], fea_2=['slot_id'])

        reduce_mem(pd.concat([adv_background_df, user_background_df], axis=1)). \
            to_pickle(self.feature_path_prefix + '{}th_pair_continue_no_clk.pkl'.format(day))

    def pair_continue_no_click(self, train, data, fea_1, fea_2):
        dense_names = []
        for i in fea_1:
            for j in fea_2:
                feature_pair = [i, j]
                no_click_feature_names = ['_'.join(feature_pair + ['continue_no_click', x]) for x in
                                          ['max', 'mean', 'var']]
                dense_names += no_click_feature_names
                tmp_df = data.groupby(feature_pair, as_index=False, sort=False)['label'].agg({'col': get_no_click_list})
                no_click_feature_df = pd.DataFrame(tmp_df['col'].tolist(), columns=no_click_feature_names)
                no_click_feature_df[feature_pair] = tmp_df[feature_pair]
                train = pd.merge(train, no_click_feature_df, on=feature_pair, how='left')
        return train[dense_names]

    def generate_clk_time_interval_feature(self, train, data, day):
        # 上次点击(不包括今天)距离现在的时间间隔，以及过去点击间隔的均值等统计信息
        # 记得排序
        suffix = 'clk'
        data = data.loc[data.label == 1].copy()
        interval_file_name = self.feature_path_prefix + '{}th_interval'.format(day) + '_' + suffix + '.pkl'
        all_features = []
        most_important_feature = ['uid', 'task_id', 'adv_id']
        interval_feature = ['uid', 'task_id', 'adv_id', 'city', 'adv_prim_id', 'device_name', 'dev_id', 'device_size',
                            'spread_app_id', 'age', 'indu_name']
        for click_feature_name in interval_feature:
            if click_feature_name in most_important_feature:
                tmp_name = ['mean_interval', 'max_interval', 'var_interval']
                pandas_func_name = ['mean', 'max', 'var']
            else:
                pandas_func_name = ['mean', 'var']
                tmp_name = ['mean_interval', 'var_interval']
            statistic_feature_name = ['_'.join([click_feature_name, feature_name, suffix]) for feature_name in
                                      tmp_name]

            last_click_day = data.groupby(click_feature_name, as_index=False, sort=False)['pt_d'].last().rename(
                columns={'pt_d': 'last_click_day'})
            previous_interval = data.groupby(click_feature_name, as_index=True, sort=False)['pt_d'].diff()
            data['previous_interval_clk'] = previous_interval
            statistic_features = data.groupby(click_feature_name, as_index=False, sort=False)[
                'previous_interval_clk'].agg(dict(zip(statistic_feature_name, pandas_func_name)))

            if click_feature_name == 'uid':
                # 添加距今为止的最后一天
                statistic_features = pd.merge(statistic_features, last_click_day, on=click_feature_name, how='left')
                statistic_features[click_feature_name + '_last_click_interval'] = day - statistic_features[
                    'last_click_day']
                statistic_features.drop(columns=['last_click_day'], inplace=True)
                all_features += [click_feature_name + '_last_click_interval']

            all_features += statistic_feature_name
            train = pd.merge(train, statistic_features, on=click_feature_name, how='left')

        reduce_mem(train[all_features]).to_pickle(interval_file_name)

    def generate_pair_clk_time_interval_feature(self, train, data, day):
        data = data.loc[data.label == 1].copy()
        user_feature = ['city', 'age', 'device_size', 'device_name']
        adv_feature = ['task_id', 'adv_id', 'adv_prim_id', 'dev_id', 'spread_app_id', 'indu_name']

        adv_background_df = self.pair_clk_interval_feature(train, data, fea_1=adv_feature, fea_2=['slot_id'])
        user_background_df = self.pair_clk_interval_feature(train, data, fea_1=user_feature + ['uid'],
                                                            fea_2=['slot_id'])
        reduce_mem(pd.concat([adv_background_df, user_background_df], axis=1)).to_pickle(
            self.feature_path_prefix + '{}th_pair_interval_clk.pkl'.format(day))

    def pair_clk_interval_feature(self, train, data, fea_1, fea_2):
        pandas_func_name = ['mean', 'var']
        tmp_name = ['mean_interval', 'var_interval']
        all_features = []
        for i in fea_1:
            for j in fea_2:
                feature_pair = [i, j]
                statistic_feature_name = ['_'.join(feature_pair + [feature_name, 'clk']) for feature_name in
                                          tmp_name]
                previous_interval = data.groupby(feature_pair, as_index=True, sort=False)['pt_d'].diff()
                data['previous_interval_clk'] = previous_interval
                statistic_features = data.groupby(feature_pair, as_index=False, sort=False)[
                    'previous_interval_clk'].agg(dict(zip(statistic_feature_name, pandas_func_name)))
                all_features += statistic_feature_name
                train = pd.merge(train, statistic_features, on=feature_pair, how='left')
        return train[all_features]

    def generate_ctr_cnt_rank_feature(self, train, data, day):
        # 点击，曝光，点击率三者排名
        for feature_name in ['uid', 'task_id']:
            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg(
                {feature_name + '_clk': 'sum', feature_name + '_cnt': 'count'})
            tmp_df[feature_name + '_ctr'] = walson_ctr(tmp_df[feature_name + '_clk'].to_numpy(),
                                                       tmp_df[feature_name + '_cnt'].to_numpy()).astype('float32')
            train = pd.merge(train, tmp_df, on=feature_name, how='left')
            # 没有历史点击的用户设置为当前有点击率用户的均值，这样防止点击率为0的用户排到第一位
            train[feature_name + '_ctr'].fillna(train[feature_name + '_ctr'].mean(), inplace=True)
            # 曝光次数缺失值填充为0
            train[feature_name + '_cnt'].fillna(0, inplace=True)
            # 点击量缺失值填充为当前众数
            train[feature_name + '_clk'].fillna(train[feature_name + '_clk'].mode()[0], inplace=True)

        feature_names = []
        # 用户点击率在其他特征下的排名
        uid_rank_list = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                         'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city',
                         'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence',
                         'his_app_size', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                         'communication_avgonline_30d', 'indu_name']

        for feature_name in uid_rank_list:
            feature_names.append(feature_name + '_uid_ctr_rank')
            feature_names.append(feature_name + '_uid_cnt_rank')
            feature_names.append(feature_name + '_uid_clk_rank')
            train[feature_name + '_uid_ctr_rank'] = train.groupby(by=feature_name)['uid_ctr'].rank(ascending=False,
                                                                                                   method='dense')
            train[feature_name + '_uid_cnt_rank'] = train.groupby(by=feature_name)['uid_cnt'].rank(ascending=False,
                                                                                                   method='dense')
            train[feature_name + '_uid_clk_rank'] = train.groupby(by=feature_name)['uid_clk'].rank(ascending=False,
                                                                                                   method='dense')
        # 广告点击率在其他特征下的排名
        task_rank_list = ['uid', 'dev_id', 'slot_id', 'app_first_class', 'age', 'city',
                          'device_name', 'device_size', 'gender', 'net_type', 'residence', 'career',
                          'his_app_size', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                          'communication_avgonline_30d', 'indu_name']
        for feature_name in task_rank_list:
            feature_names.append(feature_name + '_task_id_ctr_rank')
            feature_names.append(feature_name + '_task_id_cnt_rank')
            feature_names.append(feature_name + '_task_id_clk_rank')
            train[feature_name + '_task_id_ctr_rank'] = train.groupby(by=feature_name)['task_id_ctr'].rank(
                ascending=False,
                method='dense')
            train[feature_name + '_task_id_cnt_rank'] = train.groupby(by=feature_name)['task_id_cnt'].rank(
                ascending=False,
                method='dense')
            train[feature_name + '_task_id_clk_rank'] = train.groupby(by=feature_name)['task_id_clk'].rank(
                ascending=False,
                method='dense')
        reduce_mem(train[feature_names]).to_pickle(self.feature_path_prefix + '{}th_ctr_cnt_clk_rank.pkl'.format(day))

    def generate_today_cnt_rank_feature(self, train, data, day):
        # 点击，曝光，点击率三者排名
        for feature_name in ['uid', 'task_id']:
            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg(
                {feature_name + '_cnt': 'count'})
            train = pd.merge(train, tmp_df, on=feature_name, how='left')
            # 曝光次数缺失值填充为0
            train[feature_name + '_cnt'].fillna(0, inplace=True)

        feature_names = []
        # 用户曝光在其他特征下的排名
        uid_rank_list = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                         'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city',
                         'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence',
                         'his_app_size', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                         'communication_avgonline_30d', 'indu_name']

        for feature_name in uid_rank_list:
            feature_names.append(feature_name + '_today_uid_cnt_rank')
            train[feature_name + '_today_uid_cnt_rank'] = train.groupby(by=feature_name)['uid_cnt'].rank(
                ascending=False,
                method='dense')
        # 广告曝光在其他特征下的排名
        task_rank_list = ['uid', 'dev_id', 'slot_id', 'app_first_class', 'age', 'city',
                          'device_name', 'device_size', 'gender', 'net_type', 'residence', 'career',
                          'his_app_size', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                          'communication_avgonline_30d', 'indu_name']
        for feature_name in task_rank_list:
            feature_names.append(feature_name + '_today_task_id_cnt_rank')
            train[feature_name + '_today_task_id_cnt_rank'] = train.groupby(by=feature_name)['task_id_cnt'].rank(
                ascending=False,
                method='dense')
        reduce_mem(train[feature_names]).to_pickle(self.feature_path_prefix + '{}th_today_cnt_rank.pkl'.format(day))

    def generate_today_statistics_cnt_feature(self, train, data, day):
        # 这个特征nunique_0缺失较多
        uid_statistics_ls = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                             'slot_id', 'spread_app_id', 'tags', 'app_second_class']
        feature_df = pd.DataFrame()
        for feature in uid_statistics_ls:
            feature_pair = ['uid', feature]
            col_name = "cnt_{}_nunique_0_today".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[0], as_index=True, sort=False)[feature_pair[1]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[0], how='left')[col_name]
            col_name = "cnt_{}_nunique_1_today".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[1], as_index=True, sort=False)[feature_pair[0]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[1], how='left')[col_name]

        task_statistics_ls = ['slot_id', 'device_name', 'device_size', 'net_type',
                              'residence', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                              'communication_avgonline_30d', 'indu_name', 'age', 'city']
        for feature in task_statistics_ls:
            feature_pair = ['task_id', feature]
            col_name = "cnt_{}_nunique_0_today".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[0], as_index=True, sort=False)[feature_pair[1]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[0], how='left')[col_name]
            col_name = "cnt_{}_nunique_1_today".format("_".join(feature_pair))
            tmp_df = data[feature_pair].groupby(feature_pair[1], as_index=True, sort=False)[feature_pair[0]].agg(
                {col_name: 'nunique'})
            feature_df[col_name] = pd.merge(train, tmp_df, on=feature_pair[1], how='left')[col_name]

        reduce_mem(feature_df).to_pickle(self.feature_path_prefix + '{}th_today_statistics_cnt.pkl'.format(day))

    def generate_lastday_ctr_cnt_rank_feature(self, train, data, day):
        # 先获取点击率
        for feature_name in ['uid', 'task_id']:
            tmp_df = data.groupby(feature_name, as_index=False, sort=False)['label'].agg(
                {feature_name + '_lastday_clk': 'sum', feature_name + '_lastday_cnt': 'count'})
            tmp_df[feature_name + '_lastday_ctr'] = walson_ctr(tmp_df[feature_name + '_lastday_clk'].to_numpy(),
                                                               tmp_df[feature_name + '_lastday_cnt'].to_numpy()).astype(
                'float32')
            # 考虑是否将点击率为0的特征置为None，或者将None的置为均值
            train = pd.merge(train, tmp_df, on=feature_name, how='left')
            # 没有历史点击的用户设置为当前有点击率用户的均值，这样防止点击率为0的用户排到第一位
            train[feature_name + '_lastday_ctr'].fillna(train[feature_name + '_lastday_ctr'].mean(), inplace=True)
            # 曝光次数缺失值填充为0
            train[feature_name + '_lastday_cnt'].fillna(0, inplace=True)
            # 点击量缺失值填充为当前众数
            train[feature_name + '_lastday_clk'].fillna(train[feature_name + '_lastday_clk'].mode()[0], inplace=True)

        feature_names = []
        # 用户点击率在其他特征下的排名
        uid_rank_list = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                         'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city',
                         'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence',
                         'his_app_size', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                         'communication_avgonline_30d', 'indu_name']
        for feature_name in uid_rank_list:
            feature_names.append(feature_name + '_uid_ctr_lastday_rank')
            feature_names.append(feature_name + '_uid_cnt_lastday_rank')
            feature_names.append(feature_name + '_uid_clk_lastday_rank')
            train[feature_name + '_uid_ctr_lastday_rank'] = train.groupby(by=feature_name)['uid_lastday_ctr'].rank(
                ascending=False,
                method='dense')
            train[feature_name + '_uid_cnt_lastday_rank'] = train.groupby(by=feature_name)['uid_lastday_cnt'].rank(
                ascending=False,
                method='dense')
            train[feature_name + '_uid_clk_lastday_rank'] = train.groupby(by=feature_name)['uid_lastday_clk'].rank(
                ascending=False,
                method='dense')
        # 广告点击率在其他特征下的排名
        task_rank_list = ['uid', 'dev_id', 'slot_id', 'app_first_class', 'age', 'city',
                          'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence',
                          'his_app_size', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                          'communication_avgonline_30d', 'indu_name']
        for feature_name in task_rank_list:
            feature_names.append(feature_name + '_task_id_ctr_lastday_rank')
            feature_names.append(feature_name + '_task_id_cnt_lastday_rank')
            feature_names.append(feature_name + '_task_id_clk_lastday_rank')
            train[feature_name + '_task_id_ctr_lastday_rank'] = train.groupby(by=feature_name)[
                'task_id_lastday_ctr'].rank(
                ascending=False,
                method='dense')
            train[feature_name + '_task_id_cnt_lastday_rank'] = train.groupby(by=feature_name)[
                'task_id_lastday_cnt'].rank(
                ascending=False,
                method='dense')
            train[feature_name + '_task_id_clk_lastday_rank'] = train.groupby(by=feature_name)[
                'task_id_lastday_clk'].rank(
                ascending=False,
                method='dense')
        reduce_mem(train[feature_names]).to_pickle(
            self.feature_path_prefix + '{}th_lastday_ctr_cnt_clk_rank.pkl'.format(day))

    def generate_cnt_time_interval_feature(self, train, data, day):
        suffix = 'cnt'
        # 曝光时间差的均值、最大值、方差
        interval_file_name = self.feature_path_prefix + '{}th_interval'.format(day) + '_' + suffix + '.pkl'
        all_feature_df = []
        most_important_feature = ['uid', 'task_id', 'adv_id']
        interval_features = ['uid', 'task_id', 'adv_id', 'city', 'adv_prim_id', 'device_size', 'device_name']
        for click_feature_name in interval_features:
            if click_feature_name in most_important_feature:
                tmp_name = ['mean_interval', 'max_interval', 'var_interval']
                pandas_func_name = ['mean', 'max', 'var']
            else:
                pandas_func_name = ['mean', 'var']
                tmp_name = ['mean_interval', 'var_interval']

            statistic_feature_name = ['_'.join([click_feature_name, feature_name, suffix]) for feature_name in
                                      tmp_name]

            day_groups = data.groupby(click_feature_name, as_index=True, sort=False)['pt_d']
            previous_interval = day_groups.diff()
            data['previous_interval'] = previous_interval

            statistic_features = data.groupby(click_feature_name, as_index=False, sort=False)['previous_interval']. \
                agg(dict(zip(statistic_feature_name, pandas_func_name)))

            train_interval_statistic_features = pd.merge(train, statistic_features, on=click_feature_name, how='left')[
                statistic_feature_name]
            train_interval_statistic_features.index = train.index

            all_feature_df.append(train_interval_statistic_features)

        reduce_mem(pd.concat(all_feature_df, axis=1).reset_index(drop=True)).to_pickle(interval_file_name)

    def generate_pair_cnt_time_interval_feature(self, train, data, day):
        # 交互特征暂时只考虑用户id和广告id特征种类较少的特征交互时间间隔, 也可考虑其他特征交互间隔
        # 记得排序
        suffix = 'cnt'
        all_interval_features = []
        interval_file_name = self.feature_path_prefix + '{}th_cnt_pair_interval'.format(day) + '_' + suffix + '.csv'
        for cross_feature_name in ['slot_id', 'tags', 'creat_type_cd', 'app_second_class', 'net_type']:
            keys = ['uid', cross_feature_name]
            interval_feature_name = ['_'.join(keys) + '_' + feature_name + '_' + suffix for feature_name in
                                     ['previous_interval']]
            statistic_feature_name = ['_'.join(keys) + '_' + feature_name + '_' + suffix for feature_name in
                                      ['mean_interval', 'max_interval', 'var_interval']]
            pandas_func_name = ['mean', 'max', 'var']
            # sort一定要为false
            day_groups = data.groupby(keys, as_index=True, sort=False)['pt_d']
            previous_interval = day_groups.diff()
            data[interval_feature_name[0]] = previous_interval
            statistic_features = data.groupby(keys, as_index=False, sort=False)[interval_feature_name[0]].agg(
                dict(zip(statistic_feature_name, pandas_func_name)))
            train_interval_statistic_features = pd.merge(train, statistic_features, on=keys, how='left')[
                statistic_feature_name]
            train_interval_statistic_features.index = train.index
            train_interval_features = data[interval_feature_name].reindex(train.index)

            all_interval_features.append(train_interval_features)
            all_interval_features.append(train_interval_statistic_features)

            data.drop(columns=interval_feature_name, inplace=True)

        pd.concat(all_interval_features, axis=1).to_csv(interval_file_name, index=None)

    def trend_f(self, data, item):
        tmp = data.groupby([item, 'pt_d'], as_index=False)['label'].agg({'clk': 'sum', 'cnt': 'count'})
        features = []
        for key, df in tmp.groupby(item, as_index=False):
            feature = {}
            feature[item] = key
            for index, row in df.iterrows():
                feature[item + 'clk' + str(int(row['pt_d']))] = row['clk']
                feature[item + 'cnt' + str(int(row['pt_d']))] = row['cnt']
            features.append(feature)
        features = pd.DataFrame(features)
        return features


if __name__ == "__main__":
    stime = time.time()
    gf = GenerateFeatures()
    etime = time.time()
    print(etime - stime)
