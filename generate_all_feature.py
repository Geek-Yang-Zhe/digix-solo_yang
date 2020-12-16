import pandas as pd
from get_feature_names import get_sparse_feature_names, get_pair_click_feature_names, get_pair_time_feature_names, \
    get_id_feature_names, get_user_feature_names, get_click_feature_names, get_user_base_feature_names, \
    get_adv_base_feature_names, get_background_feature_names, get_user_profile_feautre_names
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
from gensim.models import *
import os
import itertools
import tensorflow as tf
from ge import LINE, DeepWalk
import networkx as nx
import pickle
from reduce_mem import reduce_mem

sparse_feature_names = get_sparse_feature_names()
click_feature_names = get_click_feature_names()
pair_click_feature_names = get_pair_click_feature_names()
pair_time_feature_names = get_pair_time_feature_names()
id_feature_names = get_id_feature_names()
user_feature_names = get_user_feature_names()

all_prefix = './yangzhe/feature/all/'
train_1_6_prefix = './yangzhe/feature/1_6_feature_from_1_6/'


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
    ls1 = [np.max(ls), np.mean(ls), np.median(ls), np.var(ls)]
    return ls1


class GenerateFeatures(object):
    def __init__(self):
        self.generate_all_feature_from_all_days()

    def generate_all_feature_from_all_days(self):
        # 缺点是B榜数据到了需要重新跑用全局提取特征的模型
        train_sparse_df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        # train_sparse_df.loc[train_sparse_df.pt_d == 8, 'label'] = 0
        # 以下是全局特征
        # self.cross_feature_encoding()
        # self.generate_cnt_feature(data=train_sparse_df)
        # self.generate_cross_cnt_feature()
        # self.generate_statistic_cnt_feature(data=train_sparse_df)
        self.generate_w2v_uid_feature(full_data=train_sparse_df)
        # self.generate_w2v_task_feature(full_data=train_sparse_df)
        # self.generate_graph_embedding_feature(full_data=train_sparse_df, mode='DeepWalk', suffix='cnt', graph='all')

    def generate_graph_embedding_feature(self, full_data, mode, suffix, graph='day'):
        merge_features = []
        targets = ['task_id', 'adv_id']
        embedding_sizes = [8, 8]
        epochs = [1, 1]

        for target in targets:
            self.construct_model(full_data, key='uid', target=target, suffix=suffix, graph=graph)

        for i in range(len(targets)):
            tmp = self.embedding_feature(full_data, key='uid', target=targets[i], embedding_size=embedding_sizes[i],
                                         epoch=epochs[i], suffix=suffix, order='second', mode=mode, graph=graph)
            merge_features.append(['uid', tmp[0]])
            merge_features.append([targets[i], tmp[1]])

        final_feature = []
        for key, fea in merge_features:
            tmp = pd.merge(full_data[[key]], fea, how='left', on=key)
            tmp.drop(columns=[key], inplace=True)
            final_feature.append(tmp)

        df = pd.concat(final_feature, axis=1, sort=False)
        df = reduce_mem(df)
        df.to_pickle('./yangzhe/feature/all/{}_{}_{}_uid_task_adv_embedding.pkl'.format(mode, suffix, graph))

    def embedding_feature(self, full_data, key='uid', target='task_id', embedding_size=16, epoch=10, window_size=5,
                          mode='LINE', suffix='cnt', order='second', graph='day'):
        # 调用GraphEmbedding生成全局embedding
        model_path = './yangzhe/model/n2v/{}_{}_{}_{}_{}_{}.pkl'.format(mode, suffix, key, target, graph,
                                                                        embedding_size)

        if not os.path.exists(model_path):
            G = nx.read_edgelist('./yangzhe/feature/graph/{}_{}_{}_graph.csv'.format(target, suffix, graph),
                                 create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
            tf.keras.backend.clear_session()
            if mode == 'LINE':
                model = LINE(graph=G, embedding_size=embedding_size, order=order)
                model.train(batch_size=64, epochs=epoch, verbose=1)
            else:
                model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
                model.train(embed_size=embedding_size, window_size=window_size, workers=5)
            with open(model_path, 'wb') as f:
                pickle.dump(model.get_embeddings(), f)

        # LINE对应的一阶特征与二阶特征
        if order == 'all':
            embedding_size = embedding_size * 2

        # 有些target的embedding是没有学习到的，这些不不存在于这个dict中，所以embedding中没有这些target所对应的行
        with open(model_path, 'rb') as f:
            embedding_dict = pickle.load(f)

        embedding = pd.DataFrame()
        embedding[target] = embedding_dict.keys()
        embedding['embedding'] = [embedding_dict[i] for i in embedding[target].values]
        embedding[target] = embedding[target].astype(int)

        sentences = full_data[[key, target]].groupby([key])[target].agg(list)

        # 这里是根据每个用户的历史曝光target进行均值来求用户的embedding，这些target应该在embedding[target]中
        task_id_have_embedding = set(embedding[target])
        lbl = LabelEncoder()
        lbl.fit(embedding[target])
        emb_matrix = np.array([i for i in embedding['embedding'].values])
        emb_mean = []
        for idx_list in sentences.values.tolist():
            need_key = [x for x in idx_list if x in task_id_have_embedding]
            if len(need_key) == 0:
                mean = np.zeros((embedding_size,))
            else:
                index = lbl.transform(need_key)
                mean = np.mean(emb_matrix[index], axis=0)
            emb_mean.append(mean)
        emb_feature = np.asarray(emb_mean)
        mean_col = ['{}_{}(MainKEY)_{}_MEAN_Window{}_{}'.format(mode, key, target, window_size, i) for i in
                    range(embedding_size)]
        emb_feature = pd.DataFrame(emb_feature, columns=mean_col)
        emb_feature[key] = sentences.index

        # target对应的embedding矩阵也存起来
        embeddings = np.concatenate(embedding['embedding'].values).reshape(-1, embedding_size)
        embeddings = pd.DataFrame(embeddings,
                                  columns=["{}_{}_{}(MainKEY)_Window{}_{}".format(mode, key, target, window_size, i)
                                           for i in range(embedding_size)])
        embedding[embeddings.columns] = embeddings
        del embedding['embedding']

        return emb_feature.reset_index(drop=True), embedding.reset_index(drop=True)

    def construct_model(self, full_data, key, target, suffix='cnt', graph='day'):
        # 使用1-7天的数据，训练task_id的graph embedding，仅使用曝光
        # 按照key每天点击/曝光构建session，考虑不按照天来进行构图
        if suffix == 'clk':
            full_data = full_data[full_data.label == 1]
        if not os.path.exists('./yangzhe/feature/graph/{}_{}_{}_graph.csv'.format(target, suffix, graph)):
            full_data[target] = full_data[target].astype(str)
            if graph == 'day':
                all_session = full_data.groupby([key, 'pt_d'], sort=False)[target].agg('^'.join).tolist()
            else:
                all_session = full_data.groupby([key], sort=False)[target].agg('^'.join).tolist()
            full_data[target] = full_data[target].astype(int)
            all_session = [[int(i) for i in x.split('^')] for x in all_session]
            all_session = [x for x in all_session if len(x) >= 2]
            print(len(set(itertools.chain(*all_session))))
            node_pair = dict()
            '边dict，key是节点对，权重是共现次数'
            for session in all_session:
                for i in range(1, len(session)):
                    if (session[i - 1], session[i]) not in node_pair.keys():
                        node_pair[(session[i - 1], session[i])] = 1
                    else:
                        node_pair[(session[i - 1], session[i])] += 1
            in_node_list = [[x[0], x[1]] for x in node_pair.keys()]
            weight_list = node_pair.values()
            graph_df = pd.DataFrame(in_node_list)
            graph_df['weight'] = weight_list
            graph_df.to_csv('./yangzhe/feature/graph/{}_{}_{}_graph.csv'.format(target, suffix, graph), sep=' ',
                            index=False, header=False)

    def generate_w2v_uid_feature(self, full_data):
        merge_features = []
        tmp = self.w2v_id_feature(full_data, 'uid', 'task_id', embedding_size=16, workers=10)
        merge_features.append(['uid', tmp[0]])
        merge_features.append(['task_id', tmp[1]])

        tmp = self.w2v_id_feature(full_data, 'uid', 'adv_id', embedding_size=16, workers=10)
        merge_features.append(['uid', tmp[0]])
        merge_features.append(['adv_id', tmp[1]])

        tmp = self.w2v_id_feature(full_data, 'uid', 'slot_id', embedding_size=4, workers=10)
        merge_features.append(['uid', tmp[0]])
        merge_features.append(['slot_id', tmp[1]])

        final_feature = []
        for key, fea in merge_features:
            tmp = full_data[[key]].merge(fea, how='left', on=key)
            tmp.drop(columns=[key], inplace=True)
            final_feature.append(tmp)

        df = pd.concat(final_feature, axis=1)
        df = reduce_mem(df)
        df.to_pickle('./yangzhe/feature/all/w2v_uid_task_adv_slot.pkl')

    def generate_w2v_task_feature(self, full_data):
        merge_features = []
        tmp = self.w2v_id_feature(full_data, 'task_id', 'uid', embedding_size=16, workers=10)
        merge_features.append(['task_id', tmp[0]])
        merge_features.append(['uid', tmp[1]])

        tmp = self.w2v_id_feature(full_data, 'task_id', 'city', embedding_size=8, workers=10)
        merge_features.append(['task_id', tmp[0]])
        merge_features.append(['city', tmp[1]])

        # tmp = self.w2v_id_feature(full_data, 'task_id', 'device_size', embedding_size=8, workers=10)
        # merge_features.append(['task_id', tmp[0]])
        # merge_features.append(['device_size', tmp[1]])
        #
        # tmp = self.w2v_id_feature(full_data, 'task_id', 'device_name', embedding_size=8, workers=10)
        # merge_features.append(['task_id', tmp[0]])
        # merge_features.append(['device_name', tmp[1]])

        final_feature = []
        for key, fea in merge_features:
            tmp = full_data[[key]].merge(fea, how='left', on=key)
            tmp.drop(columns=[key], inplace=True)
            final_feature.append(tmp)

        df = pd.concat(final_feature, axis=1)
        df = reduce_mem(df)
        df.to_pickle('./yangzhe/feature/all/w2v_task_id_uid_city.pkl')

    def w2v_id_feature(self, df, key1, key2,
                       embedding_size=16, window_size=20, iter=10, workers=20, min_count=0):
        # 这里是对有曝光的进行embedding
        df = df[[key1, key2]]
        sentences = df[[key1, key2]].groupby([key1])[key2].agg(list)

        if os.path.exists("./yangzhe/model/w2v/{}_{}_{}_{}.model".format(key1, key2, embedding_size, window_size)):
            model = Word2Vec.load(
                "./yangzhe/model/w2v/{}_{}_{}_{}.model".format(key1, key2, embedding_size, window_size))

        # 用户曝光的广告转换成一个二维数组,列表示uid，行是task组成的list
        else:
            # 这种方式划分,倒是不存在学不到的embedding
            text = [[str(x) for x in y] for y in sentences.values.tolist()]
            model = Word2Vec(
                text,
                size=embedding_size, window=window_size,
                min_count=min_count, sg=1, seed=2020, iter=iter, workers=workers)
            model.save("./yangzhe/model/w2v/{}_{}_{}_{}.model".format(key1, key2, embedding_size, window_size))

        embedding = pd.DataFrame()
        embedding[key2] = model.wv.vocab.keys()
        embedding['embedding'] = [model.wv[i] for i in embedding[key2].values]
        embedding[key2] = embedding[key2].astype(int)

        # 一一对应
        index_dict = {key: index for index, key in enumerate(embedding[key2].tolist())}
        emb_matrix = np.array([i for i in embedding['embedding'].values])
        for key in embedding[key2]:
            a = emb_matrix[index_dict[key]]
            b = model.wv[str(key)]
            assert (a == b).all()
        emb_mean = []
        '''
        对于新来的adv/task_id，有两个策略，一个是重新训练整体的word2vec，
        另一个则对idx_list进行判断，对于新来的，也就是不存在于embedding key2中的进行None值填充，
        填充值为np.array([None * embedding_size])
        '''
        for idx_list in sentences.values.tolist():
            index = [index_dict[i] for i in idx_list]
            mean = np.mean(emb_matrix[index], axis=0)
            emb_mean.append(mean)

        emb_feature = np.asarray(emb_mean)
        mean_col = ['{}MainKEY_{}_MEAN_Window{}_{}'.format(key1, key2, window_size, i) for i in range(embedding_size)]
        emb_feature = pd.DataFrame(emb_feature,
                                   columns=mean_col)
        emb_feature[key1] = sentences.index

        embeddings = np.concatenate(embedding['embedding'].values).reshape(-1, embedding_size)
        embeddings = pd.DataFrame(embeddings,
                                  columns=["{}_{}MainKEY_Window{}_{}".format(key1, key2, window_size, i) for i in
                                           range(embedding_size)])
        embedding[embeddings.columns] = embeddings
        del embedding['embedding']

        return emb_feature.reset_index(drop=True), embedding.reset_index(drop=True)

    def generate_cross_cnt_feature(self):
        feature_df = pd.DataFrame()
        data = pd.read_pickle(all_prefix + 'all_cnt_feature.pkl')
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

        reduce_mem(feature_df).to_pickle(all_prefix + 'all_cross_cnt_feature.pkl')

    def generate_cnt_feature(self, data):
        feature_df = pd.DataFrame()
        for feature_name in sparse_feature_names:
            feature_df["{}_cnt".format(feature_name)] = data[[feature_name]].groupby(feature_name)[
                feature_name].transform('count')
        # 交叉计数
        for i in ['uid']:
            for j in get_adv_base_feature_names():
                feature_pair = [i, j]
                feature_df["cnt-2order_{}".format("_".join(feature_pair))] = data[feature_pair].groupby(feature_pair)[
                    feature_pair[0]].transform("count")
        for i in ['uid'] + get_user_base_feature_names() + get_user_profile_feautre_names():
            for j in get_background_feature_names():
                feature_pair = [i, j]
                feature_df["cnt-2order_{}".format("_".join(feature_pair))] = data[feature_pair].groupby(feature_pair)[
                    feature_pair[0]].transform("count")
        for i in get_background_feature_names():
            for j in get_adv_base_feature_names():
                feature_pair = [i, j]
                feature_df["cnt-2order_{}".format("_".join(feature_pair))] = data[feature_pair].groupby(feature_pair)[
                    feature_pair[0]].transform("count")

        feature_df = reduce_mem(feature_df)
        feature_df.to_pickle(all_prefix + 'all_cnt_feature.pkl')

    def dense_feature_to_sparse_feature(self, data, feature_names, bin_count=10):
        df = data[feature_names].copy()
        for feature in feature_names:
            dtype = str(df[feature].dtypes)
            nunique = df[feature].nunique()
            if nunique > bin_count:
                # 最多的一个值单独拿出来，其他特征等距分箱
                tmp = df[feature]
                most_value = tmp.value_counts().index[0]
                tmp[tmp == most_value] = None
                df[feature] = pd.cut(tmp, bin_count, labels=False)
                df[feature].fillna(df[feature].max() + 1, inplace=True)
                df[feature] = df[feature].astype('int32')
                print('{}型特征{}，最多的值是:{}, 分箱前有{}种不同值，分箱后有{}种不同值'.format(dtype, feature, most_value,
                                                                       nunique, df[feature].nunique()))
            else:
                # 对于小于10的特征，他们不分箱其实就是id，没有必要，所以删除
                df.pop(feature)
        df = reduce_mem(df)
        df.columns = ['cut_' + x for x in df.columns.tolist()]
        return df

    def generate_statistic_cnt_feature(self, data):
        # 每个用户点击/曝光不同广告,素材等广告特征的数量，点击率，反映了用户的兴趣广泛程度
        uid_statistics_ls = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                             'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class']
        feature_df = pd.DataFrame()
        for feature in uid_statistics_ls:
            feature_pair = ['uid', feature]
            feature_df["cnt_{}_nunique_0".format("_".join(feature_pair))] = \
                data[feature_pair].groupby(feature_pair[0])[
                    feature_pair[1]].transform('nunique')
            feature_df["cnt_{}_nunique_1".format("_".join(feature_pair))] = \
                data[feature_pair].groupby(feature_pair[1])[
                    feature_pair[0]].transform('nunique')

        task_statistics_ls = ['slot_id', 'device_name', 'device_size', 'net_type',
                              'residence', 'emui_dev', 'list_time', 'device_price', 'up_life_duration',
                              'communication_avgonline_30d', 'indu_name', 'age', 'city', 'career', 'gender',
                              'city_rank']
        for feature in task_statistics_ls:
            feature_pair = ['task_id', feature]
            feature_df["cnt_{}_nunique_0".format("_".join(feature_pair))] = \
                data[feature_pair].groupby(feature_pair[0])[
                    feature_pair[1]].transform('nunique')
            feature_df["cnt_{}_nunique_1".format("_".join(feature_pair))] = \
                data[feature_pair].groupby(feature_pair[1])[
                    feature_pair[0]].transform('nunique')

        feature_df = reduce_mem(feature_df)
        feature_df.to_pickle(all_prefix + 'all_statistics_feature.pkl')

    def generate_ctr_feature(self, data):
        # 用户点击特征交叉，这里和第七天对应
        feat_group = []
        single_click_feature = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                                'slot_id', 'spread_app_id', 'age', 'city', 'device_name', 'device_size', 'career',
                                'residence', 'his_app_size', 'emui_dev', 'list_time',
                                'up_life_duration', 'consume_purchase', 'communication_avgonline_30d', 'indu_name']
        for feature_name in single_click_feature:
            feat_group.append([feature_name])
        train = self.kfold_stats_feature(train=data, feats=feat_group, k=7)
        train.to_csv(all_prefix + 'all_single_ctr_feature.csv', index=None)

    def generate_pair_ctr_feature(self, data):
        # 用户点击特征交叉，这里和第七天对应
        feat_group = []
        for i in ['uid']:
            for j in get_adv_base_feature_names():
                feat_group.append([i, j])
        for i in ['uid'] + get_user_base_feature_names():
            for j in get_background_feature_names():
                feat_group.append([i, j])
        for i in get_background_feature_names():
            for j in get_adv_base_feature_names():
                feat_group.append([i, j])
        train = self.kfold_stats_feature(train=data, feats=feat_group, k=6)
        train.to_csv(all_prefix + 'all_pair_ctr_feature.csv', index=None)

    def kfold_stats_feature(self, train, feats, k):
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
        kfold_features = []
        for feat in feats:
            colname = '_'.join(feat + ['ctr'])
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                # 用trn_idx组点击率计算来填充val_idx组的点击率
                tmp_df = train.loc[trn_idx].groupby(feat, as_index=False, sort=False)['label'].agg(
                    {'clk': 'sum', 'cnt': 'count'})
                tmp_df[colname] = walson_ctr(tmp_df['clk'].to_numpy(), tmp_df['cnt'].to_numpy()).astype('float32')
                train.loc[val_idx, colname] = pd.merge(train.loc[val_idx, feat], tmp_df, on=feat, how='left')[colname]
            # 缺失值用全局均值填充
            global_mean = train['label'].mean()
            train[colname].fillna(global_mean, inplace=True)
        return train[kfold_features]

    def cross_feature_encoding(self):
        data = pd.read_pickle(all_prefix + 'all_embedding_sparse_feature.pkl')
        # 将重要的交叉特征进行编码,送入到DNN中学习,重点刻画用户侧面信息
        important_cross_feature = [['task_id', 'slot_id'], ['city', 'slot_id'], ['residence', 'slot_id'],
                                   ['communication_avgonline_30d', 'slot_id'], ['list_time', 'slot_id'],
                                   ['age', 'slot_id'], ['device_name', 'slot_id'], ['up_life_duration', 'slot_id'],
                                   ['tags', 'slot_id'], ['inter_type_cd', 'slot_id'], ['adv_prim_id', 'slot_id'],
                                   ['his_app_size', 'slot_id'], ['emui_dev', 'slot_id'], ['device_size', 'slot_id'],
                                   ['adv_id', 'slot_id'],
                                   ['adv_id', 'net_type'], ['city', 'net_type'], ['device_name', 'net_type'],
                                   ['residence', 'net_type'], ]
        cross_encoding_df = pd.DataFrame()
        for feature_pair in important_cross_feature:
            data[feature_pair[0]] = data[feature_pair[0]].astype(str)
            data[feature_pair[1]] = data[feature_pair[1]].astype(str)
            tmp_series = data[feature_pair[0]] + '_' + data[feature_pair[1]]
            cross_encoding_df['_'.join(feature_pair + ['encoding'])] = LabelEncoder().fit_transform(tmp_series)

        reduce_mem(cross_encoding_df).to_pickle(all_prefix + 'all_encoding_feature_pair.pkl')


if __name__ == "__main__":
    stime = time.time()
    gf = GenerateFeatures()
    etime = time.time()
    print(etime - stime)
    # tree = Tree()
