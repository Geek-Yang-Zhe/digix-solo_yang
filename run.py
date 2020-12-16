import os

# 只显示Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
from get_feature_names import get_sparse_feature_names, get_varlen_feature_names, get_dense_feature_names, \
    get_adv_base_feature_names, get_background_feature_names
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from deepctr.models import DCN, DeepFM, MLR, NFM, DIN, DIEN, FNN, PNN, WDL, xDeepFM, AutoInt, DSIN, FiBiNET, ONN
import tensorflow as tf
from collections import defaultdict
import gc
import tensorflow.keras.backend as K
import warnings
from deepctr.layers import *
import joblib

warnings.filterwarnings('ignore')

all_prefix = './yangzhe/feature/all/'
encoder_prefix = './yangzhe/encoder/'
maxlen_prefix = './yangzhe/maxlen/'

# 试一下原始encoding特征
sparse_encoder_path = encoder_prefix + 'frequency_filter_sparse_encoders.pkl'
pair_sparse_encoder_path = encoder_prefix + 'frequency_filter_pair_sparse_encoders.pkl'
varlen_encoder_path = encoder_prefix + 'train_varlen_encoders.pkl'
varlen_maxlen_path = maxlen_prefix + 'train_varlen_maxlens.pkl'

varlen_feature_names = get_varlen_feature_names()
sparse_feature_names = get_sparse_feature_names()
dense_feature_names = get_dense_feature_names()

important_cross_feature = [['task_id', 'slot_id'], ['city', 'slot_id'], ['residence', 'slot_id'],
                           ['communication_avgonline_30d', 'slot_id'], ['list_time', 'slot_id'],
                           ['age', 'slot_id'], ['device_name', 'slot_id'], ['up_life_duration', 'slot_id'],
                           ['tags', 'slot_id'], ['inter_type_cd', 'slot_id'], ['adv_prim_id', 'slot_id'],
                           ['his_app_size', 'slot_id'], ['emui_dev', 'slot_id'], ['device_size', 'slot_id'],
                           ['adv_id', 'slot_id'],
                           ['adv_id', 'net_type'], ['city', 'net_type'], ['device_name', 'net_type'],
                           ['residence', 'net_type'], ]

'''
   目前好的模型是NFM,4-6训练能达到80.6的高分(曝光次数1次算作停用词)，使用的特征严格从该天之前提取，不使用曝光，unique，
    曝光间隔，训练1个epoch, 当前只是训练了一轮,在验证集上训练多轮发现验证集效果很好，线上测试变得很差
    发现，不使用频次过滤，效果还是不错的，待测试，感觉上可以。
    
    使用了2-7天训练,模型的结果反而不如4-7天训练的结果,改成3-7天如何呢，比2-7天好，比4-7天差，所以用4-7，
    使用了5-7的训练结果，还是不如4-7。当然上面的结果都是基于频次过滤的
    我又做了直接用原始id的，目前发现4-7原始id+part大概比频次过滤+part差了一个千分点，在此基础上做了2-7天的part+w2v特征
    结果是81.14结果还可以，但是用4-7天，同样特征效果只有0.810131，这是合理的，所以先不进行频次过滤，尽可能优化原始特征
    考虑一个问题，第二天的lastday和第二天的pastday是同一天，这样特征重复，所以考虑去掉lastday特征
    
    使用从全部区间提取的特征+w2v/line embedding之后，效果反而变差，把embedding特征去掉看下效果，还是很差，所以严格遵守
    特征提取规则。把全部区间提取特征去掉，加上w2v特征，效果比只用part特征好，但加上LINE之后，效果变差。看了下LINE的特征，有小于-1的值
    考虑可能是未归一导致的，所以将w2v和LINE归一化再做一遍实验试试。归一化后效果更差，所以最后的特征考虑part部分以及w2v未归一。
    既然w2v好，那么多学习一些w2v向量，dev_id，adv_prim_id这种的都加上
    使用了多个交叉特征编码，效果好像不如仅仅用catboost中重要的交叉特征编码，结果的确是这样，那么不用交叉特征编码效果会不会好呢，可以尝试。
    目前先尝试删除缺失值较大的特征，看效果是否有提升。我把交叉特征和缺失率较大特征删了，效果0.806563，那么到底是交叉特征影响了结果，还是
    缺失率较大的特征影响了结果呢？，先考虑加上交叉特征，删除uid_task_id_pastday和uid_adv_id_pastday试一下
    
    发现总是预测结果偏向0的，提交分数越高，也就是说预测越保守的模型，效果越好，考虑使用focal，将alpha调小，默认的gamma为2，
    alpha为0.25情况下，效果不好，只有0.802，尝试将alpha调为0.1
    同时预测方差比较大，跑了几次发现每次结果都不一样，0.811.0.813，上下浮动2个千分点
    
    使用了曝光，unique，曝光间隔，加了dropout，用0填充结果反而不行，试试用全部特征加上均值填充的结果
    使用了全部特征+填充均值，效果不如只用part特征，所以对于NFM模型最终敲定，就用part特征+w2v特征，只训练一个epoch
    
    给NFM调参，先换adgrad试试，效果很差，不知道为什么效果会这么差，收敛速度特别慢
    尝试加入w2v特征，看看效果是否有提升，效果的确有所提升，那么加入graph embedding试一下，效果不好。
    尝试了下fibinet模型，4-7训练结果0.80，尝试和0.8138的NFM加权融合，结果反而下降到0.8130。
    由于没有用到全局count特征，且曝光次数1次作为停用词效果很好，那么对ID不进行频次过滤，统计全局count特征，分箱作为新的id会怎么样呢，
    感觉这种方式比频次过滤损失的信息少。效果依旧比较差
    
    考虑用4-7+历史特征训练，用1-3+全量特征训练，1-3预测结果给4-7的模型，1-3模型考虑多种，目前用了8种模型，考虑这种可能过拟合，
    所以可以先只用同类模型，也就是用NFM这一种模型学习1-3天，预测4-7，或者使用focal loss试一下，或者1-3只用embedding特征，
    不用dense特征，或者dense特征只用word2vec，因为1-3天预测结果有很多是1，或者考虑针对这些embedding不做频次过滤
    
    ————————————————————————后期改进—————————————————————
    1. 使用交叉特征曝光次数，重要是用户和其他特征交叉，填充值为众数，发现cnt填充值基本都是1，
    那么感觉加上cnt特征应该不会减少精度，先加上，测试下2-7七天，这两种特征+part+w2v+二元交叉熵
    cross_cnt不能加，尝试加上单特征的cnt
    
    2. w2v多添加几类embedding特征，目前只有slot_id，task_id以及advid，可以添加tags, dev_id， adv_prim_id的信息
    也可以groupby task_id，之后添加age，city这些信息，测试发现效果不好
    
    3. 尝试模型融合，之前融合效果总是下降，尝试过FiBiNet和NFFM，效果都不好
    
    4. 尝试修改embedding_size和lr，查看模型结果是否有提升，4，16，重点尝试下这两个。16的提交后效果不好，应该是学习不充分，
    可以多添加一些有用的交叉特征项。
    
    5. 去掉lastday特征，因为要用第二天数据的话，lastday和pastday是同一天，发现删除了lastday特征，预测结果偏向1，
    而且测试集上效果下降6个千分点。所以在这个基础上试了下focal loss，设置gamma为0，lambda为0.25，也就是让模型偏向于预测为0。
    看下效果，效果比不用focal loss去掉lastday特征要好，所以focal loss可以调试一下。但是效果不如不去除lastday特征，
    所以lastday特征必须保留。所以在添加了特征之后，出现预测值偏激进的情况，可以通过focal loss来调整，使得模型偏向于保守。
    这样的话，应该可以随便加入cnt和nunique特征，只要把alpha调小就可以，
    事实证明，去掉lastday特征，模型过于激进，所以依旧保留。
    
    6. 添加当天的statsitcs特征和当天的这当天曝光的交叉rank特征，这个待测
    
    7. 填充值改为众数，不如均值填充
    
    8. 分成三个模型，past + last和today，past和today单模型太拉跨
    
    9. 考虑图嵌入效果不好是不是因为用了有向图，无向图直接loss溢出了
    
    10. 更改batchsize大小，尝试调小，验证集效果提升，测试集效果下降
    
    11. 过两轮，第二轮学习率衰减，当第二轮没有衰减的时候效果还可以，那么减少batchsize和lr，再次拟合一个epoch
        有效！！！，顺便再减小下batchsize，大概512和小10倍的lr合适，效果差
    
    12. 尝试第一天的数据，lastday这种特征置为-1，目前第二种尝试方法，尝试第一天学习embedding。学习结果太差
    
    13. 调整了学习率，大学习率配合大batchsize，大力出奇迹，配合embedding为4的好像确实有效。
    
    14. 1天的只有稀疏来训练embedding作为2-7训练embedding的初始化，效果不好，用2-7训练完接着增量第一天数据，效果依然不好。
'''
'''
    NFM和DeepFM模型最大的区别是,它没有将交叉求和项送入最后一层nn，而是
    在得到(batch_size, 1, embedding_dim)的交叉项embedding后, flatten后把这个和dense特征拼接到一起送入神经网络
'''


def schedule(epoch, lr):
    if epoch > 0:
        return lr * 0.1
    else:
        return lr


class RUN(object):

    def __init__(self):
        # 获取每个稀疏特征词表大小，变长特征词表大小，变长特征最大长度
        self.sparse_vocab_list = self.get_vocab_len(sparse_encoder_path)
        self.sparse_vocab_list += self.get_vocab_len(pair_sparse_encoder_path)
        self.varlen_vocab_list = self.get_vocab_len(varlen_encoder_path)
        self.varlen_maxlens = self.get_varlen_maxlen(varlen_maxlen_path)

        # 输入特征列初始化
        self.sparse_feature_columns = []
        self.varlen_feature_columns = []
        self.dense_feature_columns = []
        self.dense_dimension = 0
        self.model_name = ''
        self.minimum_day = 2
        self.maximum_day = 7
        self.embedding_dim = 8
        self.lr = 0.001
        self.batchsize = 4096
        self.epoch = 1
        self.suffix = '2_7_importance_fit'
        self.test_mode = False
        self.val = False
        self.next_epoch = False
        self.continue_train = False
        self.none_feature_count = defaultdict(float)

        self.generate_varlen_features(path='all_varlen_feature')
        self.generate_varlen_features(path='test_varlen_feature')

        if self.test_mode is True:
            self.test()
        else:
            self.train()

    def get_1th_dense_feature(self, all_feature_columns):
        feature_path_1th_prefix = './yangzhe/feature/1th_feature/'

        today_cnt_df = pd.read_pickle(feature_path_1th_prefix + '1th_today_cnt.pkl')
        today_cross_cnt_df = pd.read_pickle(feature_path_1th_prefix + '1th_pair_today_cnt.pkl')

        df = pd.concat([today_cnt_df, today_cross_cnt_df], axis=1, sort=False)

        for feature_name in df.columns:
            df[feature_name] = MinMaxScaler().fit_transform(df[feature_name].to_numpy()
                                                            .reshape(-1, 1)).astype('float32')
        for feature_name in all_feature_columns:
            if feature_name not in df.columns:
                df[feature_name] = np.zeros(df.shape[0], dtype=np.float32)
        return df

    def get_pastday_dense_feature_from_day(self, day, path=None, important=True):
        feature_path_prefix = './yangzhe/feature/{}th_feature/'.format(day)
        if path is not None:
            feature_path_prefix = path

        # 这两个特征加上测试集效果就很差
        # cnt_df = pd.read_pickle(feature_path_prefix + '{}th_cnt.pkl'.format(day))
        # statistics_df = pd.read_pickle(feature_path_prefix + '{}th_statistics_cnt.pkl'.format(day))

        today_cnt_df = pd.read_pickle(feature_path_prefix + '{}th_today_cnt.pkl'.format(day))
        today_cross_cnt_df = pd.read_pickle(feature_path_prefix + '{}th_pair_today_cnt.pkl'.format(day))

        statistics_ctr_df = pd.read_pickle(feature_path_prefix + '{}th_statistics_ctr.pkl'.format(day))
        ctr_df = pd.read_pickle(feature_path_prefix + '{}th_single_ctr.pkl'.format(day))
        pair_ctr_df = pd.read_pickle(feature_path_prefix + '{}th_pair_ctr_pastday.pkl'.format(day))
        continue_no_clk_df = pd.read_pickle(feature_path_prefix + '{}th_continue_no_clk.pkl'.format(day))
        pair_continue_no_clk_df = pd.read_pickle(feature_path_prefix + '{}th_pair_continue_no_clk.pkl'.format(day))
        interval_clk_df = pd.read_pickle(feature_path_prefix + '{}th_interval_clk.pkl'.format(day))
        pair_interval_clk_df = pd.read_pickle(feature_path_prefix + '{}th_pair_interval_clk.pkl'.format(day))
        ctr_cnt_clk_rank_df = pd.read_pickle(feature_path_prefix + '{}th_ctr_cnt_clk_rank.pkl'.format(day))

        lastday_ctr_df = pd.read_pickle(feature_path_prefix + '{}th_lastday_clk.pkl'.format(day))
        lastday_pair_ctr_df = pd.read_pickle(feature_path_prefix + '{}th_pair_lastday_clk.pkl'.format(day))
        lastday_rank_df = pd.read_pickle(feature_path_prefix + '{}th_lastday_ctr_cnt_clk_rank.pkl'.format(day))

        df = pd.concat([statistics_ctr_df, ctr_df, pair_ctr_df, today_cnt_df, today_cross_cnt_df,
                        continue_no_clk_df, pair_continue_no_clk_df, interval_clk_df,
                        pair_interval_clk_df, ctr_cnt_clk_rank_df, lastday_ctr_df, lastday_pair_ctr_df,
                        lastday_rank_df], axis=1, sort=False)
        importance_index = joblib.load('./yangzhe/feature_importance/importance_index.pkl')
        if important is False:
            no_importance_index = [x for x in df.columns if x not in importance_index]
            df = df[no_importance_index]
        else:
            df = df[importance_index]

        # 考虑去掉缺失过多的特征。
        most_non_feature_1 = ['uid_slot_id_var_interval_clk', 'uid_slot_id_mean_interval_clk', 'uid_var_interval_clk',
                              'uid_mean_interval_clk', 'uid_max_interval_clk', 'uid_last_click_interval']
        for feature in most_non_feature_1:
            if feature in df.columns:
                del df[feature]

        for feature_name in df.columns:
            # 统计下特征的缺失率，缺失率特别大的特征扔掉
            # missing_value_count = df.shape[0] - df[feature_name].count()
            # self.none_feature_count[feature_name] += missing_value_count / df.shape[0]
            df[feature_name] = df[feature_name].fillna(df[feature_name].mean())
            # df[feature_name] = StandardScaler().fit_transform(df[feature_name].to_numpy()
            #                                                   .reshape(-1, 1)).astype('float32')
            df[feature_name] = MinMaxScaler().fit_transform(df[feature_name].to_numpy()
                                                            .reshape(-1, 1)).astype('float32')
        return df

    def load_dense_df(self, minimum_day=4, maximum_day=7, important=True):
        concat_dense_list = []
        for day in range(minimum_day, maximum_day + 1):
            if day != 1:
                concat_dense_list.append(self.get_pastday_dense_feature_from_day(day=day, important=important))
                print('第{}天数据已处理完成'.format(day))
                # 当天缺失率记录
                # non_rate_df = pd.DataFrame.from_dict(self.none_feature_count, orient='index',
                #                                      columns=['non_rate']).sort_values(by='non_rate', ascending=False)
                # non_rate_df.to_csv('./{}th_non_value_df.csv'.format(day))
                # self.none_feature_count = defaultdict(float)
        if minimum_day == 1:
            if os.path.exists('./yangzhe/feature/concat_feature/1th_df.pkl'):
                df_1th = pd.read_pickle('./yangzhe/feature/concat_feature/1th_df.pkl')
            else:
                df_1th = self.get_1th_dense_feature(concat_dense_list[0].columns)
                df_1th.to_pickle('./yangzhe/feature/concat_feature/1th_df.pkl')
            concat_dense_list.insert(0, df_1th)

        part_dense_df = pd.concat(concat_dense_list, sort=False, ignore_index=True)
        del concat_dense_list
        print('已经清理掉的数量 {}'.format(gc.collect()))

        all_w2v_feature = pd.read_pickle('./yangzhe/feature/all/w2v_uid_task_adv_slot.pkl')
        sparse_df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        need_index = sparse_df[(sparse_df.pt_d >= minimum_day) & (sparse_df.pt_d <= maximum_day)].index
        w2v_df = all_w2v_feature.loc[need_index].reset_index(drop=True)

        if self.minimum_day == 4:
            all_user = pd.read_pickle('./yangzhe/graph/metapath2vec/user_df.pkl')
            user_df = all_user.loc[need_index].reset_index(drop=True)
            df = pd.concat([part_dense_df, w2v_df, user_df], axis=1, sort=False)
        else:
            df = pd.concat([part_dense_df, w2v_df], axis=1, sort=False)
        del part_dense_df, all_w2v_feature
        print('已经清理掉的数量 {}'.format(gc.collect()))
        return df

    # minimum_day到7天的数据
    def load_sparse_varlen_data(self, minimum_day=4, maximum_day=7):
        # sparse特征分成单独的sparse特征和成对编码的sparse特征，考虑删除成对编码的sprase特征
        single_sparse_df = pd.read_pickle(all_prefix + 'all_embedding_sparse_feature_with_frequency_filter.pkl')
        pair_sparse_df = pd.read_pickle(all_prefix + 'all_pair_embedding_sparse_feature_with_frequency_filter.pkl')
        sparse_df = pd.concat([single_sparse_df, pair_sparse_df], axis=1, sort=False)
        sparse_df = sparse_df[(sparse_df.pt_d >= minimum_day) & (sparse_df.pt_d <= maximum_day)]

        need_index = sparse_df.index
        sparse_df = sparse_df.reset_index(drop=True)
        del single_sparse_df, pair_sparse_df

        # 加载变长特征，每个varlen_feature，对应list中的一个元素
        all_varlen_feature_list = joblib.load(all_prefix + 'all_varlen_feature_list.pkl')
        varlen_dict = {}
        for i, feature_name in enumerate(varlen_feature_names):
            varlen_dict[feature_name] = all_varlen_feature_list[i][need_index]

        return sparse_df, varlen_dict

    def train(self):
        train_sparse_df, train_varlen_dict = self.load_sparse_varlen_data(minimum_day=self.minimum_day,
                                                                          maximum_day=self.maximum_day)
        data_name = './yangzhe/feature/all/{}_{}_importance.pkl'.format(self.minimum_day, self.maximum_day)
        if os.path.exists(data_name):
            train_dense_df = joblib.load(data_name)
        else:
            train_dense_df = self.load_dense_df(minimum_day=self.minimum_day, maximum_day=self.maximum_day, important=True)
            joblib.dump(train_dense_df, data_name)

        # 注意这里不能包括pt_d和label两个sparse_column
        self.sparse_feature_columns = [feature for feature in train_sparse_df.columns if
                                       feature not in ['label', 'pt_d']]
        self.varlen_feature_columns = varlen_feature_names
        self.dense_dimension = train_dense_df.shape[1]
        print('dense特征shape: {}, sparse特征(包括pt_d, label): {}'.format(train_dense_df.shape, train_sparse_df.shape))

        train_input_list = []
        train_input_list.extend([train_sparse_df[feature] for feature in self.sparse_feature_columns])
        train_input_list.extend([train_varlen_dict[feature] for feature in self.varlen_feature_columns])
        train_input_list.extend([train_dense_df])

        datasize = train_sparse_df.shape[0]
        steps_per_epoch = (datasize - 1) // self.batchsize + 1
        index = list(range(datasize))

        # 有一个方法是从硬盘交互，也就是把文件分块存到内存中，需要的时候读取该batch，想都不用想，速度肯定特慢
        def get_batch_train():
            while True:
                shuffle_index = np.random.permutation(index)
                for step in range(steps_per_epoch):
                    batch_index = shuffle_index[step * self.batchsize: min((step + 1) * self.batchsize, datasize)]
                    batch_input_list = []
                    batch_input_list.extend(
                        [train_sparse_df[feature].loc[batch_index] for feature in self.sparse_feature_columns])
                    batch_input_list.extend(
                        [train_varlen_dict[feature][batch_index] for feature in self.varlen_feature_columns])
                    batch_input_list.extend([train_dense_df.loc[batch_index]])
                    batch_input_label = train_sparse_df['label'].loc[batch_index]
                    yield batch_input_list, batch_input_label

        gc.collect()
        model_prefix = './yangzhe/model/all/'
        for self.model_name in ['NFM']:
            for lr in [1e-3, 3e-3]:
                for embedding_dim in [8]:
                    self.lr = lr
                    self.embedding_dim = embedding_dim
                    feature_columns = self.generate_need_same_embedding_feature_columns()
                    model, _ = self.generate_model(feature_columns)
                    model.fit(train_input_list, train_sparse_df['label'], batch_size=self.batchsize,
                              epochs=self.epoch, verbose=1)
                    # model.fit_generator(get_batch_train(),
                    #                     steps_per_epoch=steps_per_epoch, shuffle=False, epochs=self.epoch)
                    model.save(model_prefix + '{}_{}_lr{}_embedding{}.h5'.format(self.model_name, self.suffix,
                                                                                 self.lr, self.embedding_dim))
                    tf.keras.backend.clear_session()

    # 加上第一天的数据
    def continue_train_on_1th_day(self):
        train_sparse_df, train_varlen_dict = self.load_sparse_varlen_data(minimum_day=1,
                                                                          maximum_day=2)
        train_dense_df = self.load_dense_df(minimum_day=1, maximum_day=2)

        self.sparse_feature_columns = [feature for feature in train_sparse_df.columns if
                                       feature not in ['label', 'pt_d']]
        self.varlen_feature_columns = varlen_feature_names
        self.dense_dimension = train_dense_df.shape[1]

        train_1th_index = train_sparse_df[train_sparse_df.pt_d == 1].index
        datasize = len(train_1th_index)
        steps_per_epoch = (datasize - 1) // self.batchsize + 1
        index = list(range(datasize))
        np.random.seed(2020)
        shuffle_index = np.random.permutation(index)

        def get_batch_train():
            # 将train_1th_index洗牌
            for step in range(steps_per_epoch):
                batch_index = shuffle_index[step * self.batchsize: min((step + 1) * self.batchsize, datasize)]
                batch_input_list = []
                batch_input_list.extend(
                    [train_sparse_df[feature].loc[batch_index] for feature in self.sparse_feature_columns])
                batch_input_list.extend(
                    [train_varlen_dict[feature][batch_index] for feature in self.varlen_feature_columns])
                batch_input_list.extend([train_dense_df.loc[batch_index]])
                batch_input_label = train_sparse_df['label'].loc[batch_index]
                yield batch_input_list, batch_input_label

        model = tf.keras.models.load_model('./yangzhe/model/all/2-7.h5',
                                           custom_objects={'SequencePoolingLayer': SequencePoolingLayer,
                                                           'NoMask': NoMask,
                                                           'BiInteractionPooling': BiInteractionPooling,
                                                           'Linear': Linear,
                                                           'DNN': DNN,
                                                           'PredictionLayer': PredictionLayer})
        model.fit_generator(get_batch_train(), steps_per_epoch=steps_per_epoch, shuffle=False, epochs=self.epoch)
        model.save('./yangzhe/model/all/1-7.h5')

    def test(self):
        # sparse特征分成单独的sparse特征和成对编码的sparse特征
        single_sparse_df = pd.read_pickle(all_prefix + 'test_embedding_sparse_feature_with_frequency_filter.pkl')
        pair_sparse_df = pd.read_pickle(all_prefix + 'test_pair_embedding_sparse_feature_with_frequency_filter.pkl')
        test_sparse_df = pd.concat([single_sparse_df, pair_sparse_df], axis=1, sort=False)

        # 加载test变长特征
        all_varlen_feature_list = joblib.load(all_prefix + 'test_varlen_feature_list.pkl')
        test_varlen_dict = {}
        for i, feature_name in enumerate(varlen_feature_names):
            test_varlen_dict[feature_name] = all_varlen_feature_list[i]

        # 将第8天和第9天归并为一天，效果好些
        part_dense_df = self.get_pastday_dense_feature_from_day(day=8, path='./yangzhe/feature/8_9th_feature/', important=True)
        # 当天缺失率记录
        # non_rate_df = pd.DataFrame.from_dict(self.none_feature_count, orient='index',
        #                                      columns=['non_rate']).sort_values(by='non_rate', ascending=False)
        # non_rate_df.to_csv('./8th_non_value_df.csv')
        # self.none_feature_count = defaultdict(float)

        # concat_dense_list = []
        # for day in range(8, 10):
        #     concat_dense_list.append(self.get_pastday_dense_feature_from_day(day=day))
        # part_dense_df = pd.concat(concat_dense_list, sort=False, ignore_index=True)

        all_w2v_feature = pd.read_pickle('./yangzhe/feature/all/w2v_uid_task_adv_slot.pkl')
        sparse_df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        need_index = sparse_df[sparse_df.pt_d >= 8].index
        w2v_df = all_w2v_feature.loc[need_index].reset_index(drop=True)

        if self.minimum_day == 4:
            all_user = pd.read_pickle('./yangzhe/graph/metapath2vec/user_df.pkl')
            user_df = all_user.loc[need_index].reset_index(drop=True)
            test_dense_df = pd.concat([part_dense_df, w2v_df, user_df], axis=1, sort=False)
        else:
            test_dense_df = pd.concat([part_dense_df, w2v_df], axis=1, sort=False)

        # del part_dense_df, all_w2v_feature
        print('已经清理掉的数量 {}'.format(gc.collect()))

        print('dense特征shape: {}, sparse特征(包括pt_d, label): {}'.format(test_dense_df.shape, test_sparse_df.shape))

        self.sparse_feature_columns = [feature for feature in test_sparse_df.columns if
                                       feature not in ['label', 'pt_d']]
        self.varlen_feature_columns = varlen_feature_names
        self.dense_dimension = test_dense_df.shape[1]

        test_input_list = []
        test_input_list.extend([test_sparse_df[feature] for feature in self.sparse_feature_columns])
        test_input_list.extend([test_varlen_dict[feature] for feature in self.varlen_feature_columns])
        test_input_list.extend([test_dense_df])

        self.model_name = 'NFM'
        model_prefix = './yangzhe/model/all/'

        # 模型组预测
        # for seed in range(2020, 2021):
        for self.model_name in ['NFM']:
            for lr in [4e-3]:
                for embedding_dim in [16]:
                    tf.keras.backend.clear_session()

                    self.lr = lr
                    self.embedding_dim = embedding_dim
                    feature_columns = self.generate_need_same_embedding_feature_columns()
                    model, _ = self.generate_model(feature_columns)
                    model_name = '{}_{}_lr{}_embedding{}'.format(self.model_name, self.suffix,
                                                                 self.lr, self.embedding_dim)
                    # model_name = '{}_{}_seed{}_lr{}_embedding{}'.format(self.model_name, self.suffix, seed,
                    #                                                     self.lr, self.embedding_dim)
                    model.load_weights(model_prefix + model_name + '.h5')

                    probabilities = model.predict(test_input_list, batch_size=self.batchsize)
                    probabilities = np.around(probabilities, decimals=6).ravel()

                    # pd.DataFrame({'id': test_sparse_df['label'], 'probability': probabilities}). \
                    #     to_csv('./yangzhe/feature/the_final_test_A/{}.csv'.format(model_name),
                    #            index=None)

                    A_index = list(range(0, 2000000))
                    B_index = list(range(2000000, 4000000))

                    id_A = test_sparse_df.loc[A_index, 'label']
                    probability_A = probabilities[A_index]

                    id_B = test_sparse_df.loc[B_index, 'label']
                    probability_B = probabilities[B_index]

                    pd.DataFrame({'id': id_A, 'probability': probability_A}). \
                        to_csv('./yangzhe/feature/final_test_A/{}.csv'.format(model_name),
                               index=None)
                    pd.DataFrame({'id': id_B, 'probability': probability_B}). \
                        to_csv('./yangzhe/feature/final_test_B/{}.csv'.format(model_name),
                               index=None)

    def generate_varlen_features(self, path='all_varlen_feature'):
        varlen_df = pd.read_csv(all_prefix + '{}.csv'.format(path), dtype=str)
        # 这部分处理非常慢，拿出来序列化存储
        varlen_features = []
        for idx, feature_name in enumerate(varlen_feature_names):
            try:
                varlen_list = [[int(i) for i in x.split('^')] for x in varlen_df[feature_name]]
                varlen_list = pad_sequences(varlen_list, maxlen=self.varlen_maxlens[idx], padding='post',
                                            truncating='post')
                varlen_features.append(varlen_list)
            except Exception:
                print(varlen_df[feature_name])
                exit(0)
        joblib.dump(varlen_features, './yangzhe/feature/all/{}_list.pkl'.format(path))
        print('变长特征处理完成')

    def generate_need_same_embedding_feature_columns(self):
        sparse_feature_columns = []
        dense_feature_columns = []
        varlen_feature_columns = []
        for i, feature_name in enumerate(self.sparse_feature_columns):
            sparse_feature_columns.append(SparseFeat(name=feature_name, vocabulary_size=self.sparse_vocab_list[i],
                                                     embedding_dim=self.embedding_dim))
        for i, feature_name in enumerate(self.varlen_feature_columns):
            varlen_feature_columns.append(VarLenSparseFeat(SparseFeat(name=feature_name,
                                                                      vocabulary_size=self.varlen_vocab_list[i],
                                                                      embedding_dim=self.embedding_dim),
                                                           maxlen=self.varlen_maxlens[i], combiner='mean'))
        if len(self.dense_feature_columns) != 0:
            for feature_name in self.dense_feature_columns:
                dense_feature_columns.append(DenseFeat(name=feature_name, dimension=1))
        else:
            dense_feature_columns.append(DenseFeat(name='all_dense_feature', dimension=self.dense_dimension))

        return sparse_feature_columns + varlen_feature_columns + dense_feature_columns

    def get_vocab_len(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            encoders = pickle.load(f)
        # 留出一位给新增元素，词表是从1开始编码的
        vocab_lens = [len(encoders[i]) + 1 for i in range(len(encoders))]
        return vocab_lens

    def get_varlen_maxlen(self, maxlen_path):
        with open(maxlen_path, 'rb') as f:
            maxlen_list = pickle.load(f)
        return maxlen_list

    def generate_model(self, columns):

        model = NFM(columns, columns)

        if self.model_name == 'DeepFM':
            model = DeepFM(columns, columns)
        if self.model_name == 'DCN':
            model = DCN(columns, columns)
        if self.model_name == 'MLR':
            model = MLR(columns)
        if self.model_name == 'FNN':
            model = FNN(columns, columns)
        if self.model_name == 'PNN':
            model = PNN(columns)
        if self.model_name == 'WDL':
            model = WDL(columns, columns)
        if self.model_name == 'xDeepFM':
            model = xDeepFM(columns, columns)
        if self.model_name == 'AutoInt':
            model = AutoInt(columns, columns)
        if self.model_name == 'FiBiNET':
            model = FiBiNET(columns, columns)

        if self.val is False:
            model_prefix = './yangzhe/model/all/'
            model_suffix = '_epoch{epoch:02d}.h5'
            if self.next_epoch is True:
                monitor = 'auc_1'
            else:
                monitor = 'auc'
            save_path = model_prefix + '{}_{}'.format(self.model_name, self.suffix) + model_suffix
            model_check_point = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor=monitor, mode='max',
                                                                   verbose=1, save_best_only=False,
                                                                   save_weights_only=True)
            # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./log/{}'.format(self.suffix),
            #                                              update_freq=10 * self.batchsize)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
            callbacks = [model_check_point, lr_scheduler]
        else:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=0, mode='max')
            callbacks = [early_stopping]

        model.compile(tf.keras.optimizers.Adam(learning_rate=self.lr), 'binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC()])
        return model, callbacks


if __name__ == "__main__":
    run = RUN()
