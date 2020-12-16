import pandas as pd
import pickle
import numpy as np
from reduce_mem import reduce_mem
from tqdm import tqdm
from get_feature_names import get_varlen_feature_names, get_sparse_feature_names, get_dense_feature_names, \
    get_background_feature_names, get_adv_base_feature_names, \
    get_user_base_feature_names, get_user_profile_feautre_names

'''
     统计了下用户分布,发现在9成用户在4-7这个区间有记录,大约105.6w, 5-7这个区间大概99.5w, 
     所以特征提取1-3/4比较合适,4/5-7用来训练, 感觉对用户分开建模,对测试集中1-3天的用户采用1-3训练的模型预测也行
     注意了,这里训练集不是按照天数排好序的
     顺便统计下测试集中用户最后一次曝光的分布情况,可以看到在7,6,5,4分布较多,1,2,3分布较少,考虑分开建模
        7.0    706176
        NaN    211540
        6.0     49002
        5.0     17166
        4.0      7166
        3.0      4083
        2.0      2850
        1.0      2017
        
     loss小的线上效果好,这说明模型越偏向预测为0的,效果可能越好.
     之前在停用词的处理上,是只将出现1次的作为停用词,感觉可以考虑只将曝光未点击的, 出现1次的作为停用词
     无论怎么样NFM貌似都是一个不错的选择,但如果在停用词中删除有点击的,效果会非常差,所以需要连带着曝光即点击一起参与编码
     算一下用户每天平均曝光次数,应该为5次,考虑去除曝光次数小于2次的用户,也就是曝光次数1-2当成新用户,3-4不活跃用户,
     测试结果发现曝光次数2次的效果不好,还是调整到1次
'''

# 特征分为三种，稀疏、密集、变长特征
varlen_feature_names = get_varlen_feature_names()
sparse_feature_names = get_sparse_feature_names()
dense_feature_names = get_dense_feature_names()

train_read_path = '../final/train_data.csv'
test_A_read_path = '../final/test_data_A.csv'
test_B_read_path = '../final/test_data_B.csv'

sparse_encoder_path = './yangzhe/encoder/frequency_filter_sparse_encoders.pkl'
pair_sparse_encoder_path = './yangzhe/encoder/frequency_filter_pair_sparse_encoders.pkl'
varlen_encoder_path = './yangzhe/encoder/train_varlen_encoders.pkl'
varlen_maxlen_path = './yangzhe/maxlen/train_varlen_maxlens.pkl'

# 这里可以构造更多的交叉特征
important_cross_feature = [['task_id', 'slot_id'], ['city', 'slot_id'], ['residence', 'slot_id'],
                           ['communication_avgonline_30d', 'slot_id'], ['list_time', 'slot_id'],
                           ['age', 'slot_id'], ['device_name', 'slot_id'], ['up_life_duration', 'slot_id'],
                           ['tags', 'slot_id'], ['inter_type_cd', 'slot_id'], ['adv_prim_id', 'slot_id'],
                           ['his_app_size', 'slot_id'], ['emui_dev', 'slot_id'], ['device_size', 'slot_id'],
                           ['adv_id', 'slot_id'],
                           ['adv_id', 'net_type'], ['city', 'net_type'], ['device_name', 'net_type'],
                           ['residence', 'net_type'], ]


# get_background一定要留，其他特征可以考虑削减
# important_cross_feature = [[i, j] for i in get_adv_base_feature_names() for j in get_background_feature_names()] + \
#                           [[i, j] for i in get_user_base_feature_names() for j in get_background_feature_names()] + \
#                           [[i, j] for i in get_user_profile_feautre_names() for j in get_background_feature_names()] + \
#                           [[i, j] for i in get_adv_base_feature_names() for j in get_user_base_feature_names()]

# important_cross_feature = [[i, j] for i in get_adv_base_feature_names() for j in get_background_feature_names()] + \
#                           [[i, j] for i in get_user_base_feature_names() for j in get_background_feature_names()] + \
#                           [[i, j] for i in get_adv_base_feature_names() for j in get_user_profile_feautre_names()] + \
#                           [[i, j] for i in get_user_base_feature_names() for j in get_user_profile_feautre_names()]


class ProcessTrainData(object):

    def __init__(self):
        self.process_feature()

    def process_feature(self):
        # 对于训练集,归并排序按照天数排好
        train_df = pd.read_csv(train_read_path, sep='|')
        train_df = train_df.sort_values('pt_d', kind='mergesort').reset_index(drop=True)

        # B榜到了把B榜和test_df拼接到一起
        test_df_A = pd.read_csv(test_A_read_path, sep='|')
        test_df_B = pd.read_csv(test_B_read_path, sep='|')
        test_df = pd.concat([test_df_A, test_df_B], axis=0, ignore_index=True, sort=False)
        test_df = test_df.rename(columns={'id': 'label'})

        # 对于变长特征的编码直接编码就行，不做频次过滤
        self.process_varlen_feature(train_df, varlen_encoder_path, varlen_maxlen_path)
        self.process_B_varlen_feature(test_df)
        df = pd.concat([train_df[varlen_feature_names], test_df[varlen_feature_names]], ignore_index=True, sort=False)
        df.to_csv('./yangzhe/feature/all/all_varlen_feature.csv', index=None)
        test_df[varlen_feature_names].to_csv('./yangzhe/feature/all/test_varlen_feature.csv', index=None)
        # self.process_varlen_dense_feature(df)

        # 原始特征存储，去掉变长特征
        df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        reduce_mem(df[sparse_feature_names + ['label', 'pt_d']]).to_pickle('./yangzhe/feature/all/all_feature.pkl')

        # 交叉特征编码，解码
        train_pair_encoder_df = self.process_sparse_feature_with_frequency_filter(train_df, pair_sparse_encoder_path,
                                                                                  important_cross_feature, pair=True)
        test_pair_encoder_df = self.process_B_sparse_feature(test_df, pair_sparse_encoder_path, important_cross_feature,
                                                             pair=True)
        reduce_mem(test_pair_encoder_df).to_pickle(
            './yangzhe/feature/all/test_pair_embedding_sparse_feature_with_frequency_filter.pkl')
        df = pd.concat([train_pair_encoder_df, test_pair_encoder_df], ignore_index=True, sort=False)
        reduce_mem(df).to_pickle('./yangzhe/feature/all/all_pair_embedding_sparse_feature_with_frequency_filter.pkl')

        # 单特征编码，频次过滤
        train_encoder_df = self.process_sparse_feature_with_frequency_filter(train_df, sparse_encoder_path,
                                                                             sparse_feature_names)
        test_encoder_df = self.process_B_sparse_feature(test_df, sparse_encoder_path, sparse_feature_names)
        train_encoder_df['label'] = train_df['label']
        train_encoder_df['pt_d'] = train_df['pt_d']
        test_encoder_df['label'] = test_df['label']
        test_encoder_df['pt_d'] = test_df['pt_d']
        reduce_mem(test_encoder_df[sparse_feature_names + ['label', 'pt_d']]).to_pickle(
            './yangzhe/feature/all/test_embedding_sparse_feature_with_frequency_filter.pkl')
        df = pd.concat([train_encoder_df, test_encoder_df], ignore_index=True, sort=False)
        reduce_mem(df).to_pickle('./yangzhe/feature/all/all_embedding_sparse_feature_with_frequency_filter.pkl')

    def process_original_feature(self):
        data = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        train_df = data[data.pt_d <= 7]
        test_df = data[data.pt_d == 8]

        # 交叉特征编码,不使用停用词
        train_encoder_df = self.process_sparse_feature_without_frequency_filter(train_df, pair_sparse_encoder_path,
                                                                                important_cross_feature, pair=True)
        test_encoder_df = self.process_B_sparse_feature(test_df, pair_sparse_encoder_path,
                                                        important_cross_feature, pair=True)
        df = pd.concat([train_encoder_df, test_encoder_df], ignore_index=True, sort=False)
        reduce_mem(test_encoder_df).to_pickle(
            './yangzhe/feature/all/original_test_pair_embedding_sparse_feature_with_frequency_filter.pkl')
        reduce_mem(df).to_pickle('./yangzhe/feature/all/original_all_pair_embedding_sparse_feature.pkl')

        # 单特征编码，不使用停用词
        train_df = self.process_sparse_feature_without_frequency_filter(train_df, sparse_encoder_path,
                                                                        sparse_feature_names)
        test_df = self.process_B_sparse_feature(test_df, sparse_encoder_path, sparse_feature_names)
        reduce_mem(train_df[sparse_feature_names + ['label', 'pt_d']]).to_pickle(
            './yangzhe/feature/all/original_test_embedding_sparse_feature_with_frequency_filter.pkl')
        df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        reduce_mem(df).to_pickle('./yangzhe/feature/all/original_all_embedding_sparse_feature_with_frequency_filter.pkl')

    def get_cross_df(self, df):
        cross_df = pd.DataFrame()
        for feature_pair in important_cross_feature:
            cross_df['_'.join(feature_pair + ['encoding'])] = df[feature_pair[0]].astype(str) + '_' + df[
                feature_pair[1]].astype(str)
            print('_'.join(feature_pair + ['encoding']) + '特征生成完成')
        return cross_df

    def encoder(self, keys, encoder_dict):
        for key in keys:
            if key not in encoder_dict:
                # 0留给新增元素和停用词,比如新加入的ID
                encoder_dict[key] = len(encoder_dict) + 1
        return [encoder_dict[x] for x in keys]

    def decoder(self, keys, encoder_dict):
        for key in keys:
            if key not in encoder_dict:
                # 0赋值给新加入的元素,当然这里只是针对embedding才这么做
                encoder_dict[key] = 0
        return [encoder_dict[x] for x in keys]

    # 这里进行频次过滤
    def process_sparse_feature_with_frequency_filter(self, df, encoder_path, feature_names, pair=False):
        if pair is True:
            df = df.astype(str)
        encoder_df = pd.DataFrame()
        sparse_encoders = list()
        for feature_name in feature_names:
            sparse_encoder = dict()

            if pair is True:
                cross_name = '_'.join(feature_name + ['encoding'])
                encoder_df[cross_name] = df[feature_name[0]] + '_' + df[feature_name[1]]
            else:
                cross_name = feature_name
                encoder_df[cross_name] = df[feature_name]

            tmp = encoder_df[cross_name]
            counts = tmp.value_counts()
            no_need_id = counts[counts <= 1].index
            no_need_index = tmp[tmp.isin(no_need_id)].index
            need_index = tmp[~(tmp.isin(no_need_id))].index

            if len(no_need_index) != 0:
                print(cross_name, len(no_need_index))
                encoder_df.loc[no_need_index, cross_name] = 0

            encoder_df.loc[need_index, cross_name] = self.encoder(encoder_df.loc[need_index, cross_name],
                                                                  sparse_encoder)
            encoder_df[cross_name] = encoder_df[cross_name].astype('int32')

            sparse_encoders.append(sparse_encoder)
        with open(encoder_path, 'wb') as f:
            pickle.dump(sparse_encoders, f)
        return encoder_df

    def process_sparse_feature_without_frequency_filter(self, df, encoder_path, feature_names, pair=False):
        if pair is True:
            df = df.astype(str)
        encoder_df = pd.DataFrame()
        sparse_encoders = list()
        for feature_name in tqdm(feature_names):
            sparse_encoder = dict()

            # 这个不去除停用词
            if pair is True:
                cross_name = '_'.join(feature_name + ['encoding'])
                encoder_df[cross_name] = df[feature_name[0]] + '_' + df[feature_name[1]]
            else:
                cross_name = feature_name
                encoder_df[cross_name] = df[feature_name]

            encoder_df[cross_name] = self.encoder(encoder_df[cross_name], sparse_encoder)
            encoder_df[cross_name] = encoder_df[cross_name].astype('int32')

            sparse_encoders.append(sparse_encoder)
        with open(encoder_path, 'wb') as f:
            pickle.dump(sparse_encoders, f)
        return encoder_df

    def process_B_sparse_feature(self, df, decoder_path, feature_names, pair=False):
        if pair is True:
            df = df.astype(str)
        encoder_df = pd.DataFrame()
        with open(decoder_path, 'rb') as f:
            sparse_encoders = pickle.load(f)
        for i, feature_name in tqdm(enumerate(feature_names)):
            sparse_encoder = sparse_encoders[i]

            if pair is True:
                cross_name = '_'.join(feature_name + ['encoding'])
                encoder_df[cross_name] = df[feature_name[0]] + '_' + df[feature_name[1]]
            else:
                cross_name = feature_name
                encoder_df[cross_name] = df[feature_name]

            encoder_df[cross_name] = self.decoder(encoder_df[cross_name], sparse_encoder)
            encoder_df[cross_name] = encoder_df[cross_name].astype('int32')
        return encoder_df

    def process_varlen_feature(self, df, encoder_path, maxlen_path):
        varlen_encoders = list()
        varlen_maxlens = list()
        for feature_name in varlen_feature_names:
            varlen_encoder = dict()
            varlen_list = [x.split('^') for x in df[feature_name].values]
            # 记录最大长度, 写入df中,还是以^间隔的字符串
            maxlen = max([len(x) for x in varlen_list])
            for i, row in enumerate(varlen_list):
                varlen_list[i] = '^'.join([str(x) for x in self.encoder(row, varlen_encoder)])
            df[feature_name] = varlen_list
            varlen_encoders.append(varlen_encoder)
            varlen_maxlens.append(maxlen)
        with open(encoder_path, 'wb') as f:
            pickle.dump(varlen_encoders, f)
        with open(maxlen_path, 'wb') as f:
            pickle.dump(varlen_maxlens, f)

    def process_B_varlen_feature(self, df):
        with open(varlen_encoder_path, 'rb') as f:
            varlen_encoders = pickle.load(f)
        for i, feature_name in enumerate(varlen_feature_names):
            varlen_encoder = varlen_encoders[i]
            varlen_list = [x.split('^') for x in df[feature_name].values]
            for j, row in enumerate(varlen_list):
                varlen_list[j] = '^'.join([str(x) for x in self.decoder(row, varlen_encoder)])
            df[feature_name] = varlen_list

    def process_varlen_dense_feature(self, df):
        df_list = []
        for feature_name in tqdm(varlen_feature_names):
            varlen_list = [np.array([int(i) for i in x.split('^')]) for x in df[feature_name].values]
            tmp_df = pd.DataFrame([[x.size, x.mean(), x.var(), x.max(), x.min()] for x in varlen_list],
                                  columns=[feature_name + '_' + x for x in ['len', 'mean', 'var', 'max', 'min']])
            df_list.append(tmp_df)
        reduce_mem(pd.concat(df_list, axis=1)).to_pickle('./yangzhe/feature/all/all_varlen_dense_feature.pkl')


if __name__ == '__main__':
    procss_train = ProcessTrainData()
