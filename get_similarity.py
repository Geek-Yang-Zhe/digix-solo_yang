from get_feature_names import get_user_profile_feautre_names, get_user_base_feature_names
import pandas as pd
import numpy as np
import pickle
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from tqdm import tqdm
from reduce_mem import reduce_mem

user_feature_list = get_user_profile_feautre_names() + get_user_base_feature_names()

def save_user_feature():
    # 计算1-3天有点击的用户和其他用户之间的相似度
    all_feature = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
    click_user_1_3 = set(all_feature.loc[(all_feature['label'] == 1) & (all_feature['pt_d'] <= 3), 'uid'])
    print('1~3天有点击的用户数量为: {}'.format(len(click_user_1_3)))
    other_user = set(all_feature['uid']) - click_user_1_3
    print('其他用户数量为: {}'.format(len(other_user)))
    all_click_user_df = all_feature.loc[all_feature['uid'].isin(click_user_1_3), ['uid'] + user_feature_list]
    all_non_click_user_df = all_feature.loc[all_feature['uid'].isin(other_user), ['uid'] + user_feature_list]

    # 使用用户最后一次记录的基本特征作为用户最终的基本特征
    all_click_user_df = all_click_user_df.groupby('uid', as_index=False, sort=False).last()
    all_non_click_user_df = all_non_click_user_df.groupby('uid', as_index=False, sort=False).last()
    all_click_user_df.to_pickle('./yangzhe/graph/user_feature/all_click_user.pkl')
    all_non_click_user_df.to_pickle('./yangzhe/graph/user_feature/all_non_click_user.pkl')


def get_similar_user():
    clk_user_df = pd.read_pickle('./yangzhe/graph/user_feature/all_click_user.pkl')
    non_clk_user_df = pd.read_pickle('./yangzhe/graph/user_feature/all_non_click_user.pkl')
    clk_user_array = np.array(clk_user_df['uid'])
    non_clk_user_array = np.array(non_clk_user_df['uid'])
    clk_npy = np.asarray(clk_user_df[user_feature_list])
    non_clk_npy = np.asarray(non_clk_user_df[user_feature_list])

    def get_most_similar_user(non_clk_index):
        similarity = clk_npy.shape[1] - np.count_nonzero(non_clk_npy[non_clk_index] - clk_npy, axis=1)
        max_value = np.max(similarity)
        clk_index = np.where(similarity == max_value)[0]
        return non_clk_index, clk_index

    # 获取最相似的用户
    res = Parallel(n_jobs=36, verbose=1)(delayed(get_most_similar_user)(idx) for idx in range(non_clk_npy.shape[0]))
    similarity_user_dict = {non_clk_user_array[non_clk_index]: clk_user_array[clk_index] for non_clk_index, clk_index in res}
    with open('./yangzhe/graph/user_feature/user_similarity.pkl', 'wb') as f:
        pickle.dump(similarity_user_dict, f)
    print(similarity_user_dict)


def get_similar_item():
    clk_user_df = pd.read_pickle('./yangzhe/graph/user_feature/all_click_user.pkl')
    non_clk_user_df = pd.read_pickle('./yangzhe/graph/user_feature/all_non_click_user.pkl')
    clk_user_array = np.array(clk_user_df['uid'])
    non_clk_user_array = np.array(non_clk_user_df['uid'])
    clk_npy = np.asarray(clk_user_df[user_feature_list])
    non_clk_npy = np.asarray(non_clk_user_df[user_feature_list])

    def get_most_similar_user(non_clk_index):
        similarity = clk_npy.shape[1] - np.count_nonzero(non_clk_npy[non_clk_index] - clk_npy, axis=1)
        max_value = np.max(similarity)
        clk_index = np.where(similarity == max_value)[0]
        return non_clk_index, clk_index

    # 获取最相似的用户
    res = Parallel(n_jobs=36, verbose=1)(delayed(get_most_similar_user)(idx) for idx in range(non_clk_npy.shape[0]))
    similarity_user_dict = {non_clk_user_array[non_clk_index]: clk_user_array[clk_index] for non_clk_index, clk_index in res}
    with open('./yangzhe/graph/user_feature/user_similarity.pkl', 'wb') as f:
        pickle.dump(similarity_user_dict, f)
    print(similarity_user_dict)


def get_non_clk_user_embedding():
    with open('./yangzhe/graph/user_feature/user_similarity.pkl', 'rb') as f:
        similarity_user_dict = pickle.load(f)
    # model.wv.vocab.keys()存着uid和task_id，需要先从中分离出uid
    clk_uid = pd.read_pickle('./yangzhe/graph/user_feature/all_click_user.pkl')['uid'].tolist()
    model = Word2Vec.load('./yangzhe/graph/{}_{}emb_{}iter_{}window.model'.format('click', 16, 5, 5))
    uid_embedding_dict = {uid: model.wv[str(uid)] for uid in clk_uid}
    # index_dict和embedding_npy和中的顺序一一对应
    embedding_npy = np.asarray([uid_embedding_dict[key] for key in clk_uid])
    index_dict = {key: index for index, key in enumerate(clk_uid)}

    for key in clk_uid:
        condition = embedding_npy[index_dict[key]] == uid_embedding_dict[key]
        assert condition.all()

    # 计算non_clk_uid的embedding
    for non_clk_uid in tqdm(similarity_user_dict):
        similar_clk_uid = similarity_user_dict[non_clk_uid]
        index = [index_dict[i] for i in similar_clk_uid]
        uid_embedding_dict[non_clk_uid] = np.mean(embedding_npy[index], axis=0)

    user_df = pd.DataFrame.from_dict(uid_embedding_dict, orient='index', columns=['uid_{}'.format(i) for i in range(16)])
    user_df = user_df.reset_index().rename(columns={'index': 'uid'})
    all_user_df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')[['uid']]
    user_df = pd.merge(all_user_df, user_df, how='left', on='uid')
    assert not user_df.isna().any().any()
    del user_df['uid']
    reduce_mem(user_df).to_pickle('./yangzhe/graph/metapath2vec/user_df.pkl')
    print(user_df)


if __name__ == '__main__':
    save_user_feature()
    get_similar_user()
    get_non_clk_user_embedding()

