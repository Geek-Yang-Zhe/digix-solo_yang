import pandas as pd
from collections import OrderedDict
from get_feature_names import get_varlen_feature_names, get_sparse_feature_names, get_dense_feature_names
import joblib

sparse_feature = get_sparse_feature_names()

train_read_path = 'train_data.csv'
test_read_path = 'test_data_A.csv'

class Statistic(object):

    def get_ctr_per_day(self):
        df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        df = df[df['pt_d'] <= 7].reset_index(drop=True)
        df = df.groupby('pt_d', as_index=False)['label'].agg({'clk': 'sum','cnt':'count'})
        df['ctr'] = df['clk'] / df['cnt']
        df.to_csv('./yangzhe/statistic/day_label.csv', index=None)

    def get_user_mean_click(self):
        df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        test_user = set(df.loc[df.pt_d >= 8, 'uid'])
        df = df[df.uid.isin(test_user) & (df.pt_d <= 7)].reset_index(drop=True)
        uid_clk_df = df[['uid', 'label']].groupby('uid', as_index=False, sort=False).agg('sum')
        uid_clk_df['label'].value_counts().to_csv('./yangzhe/statistic/test_uid_clk_counts.csv')
        uid_clk_df['label'].describe().to_csv('./yangzhe/statistic/test_uid_clk.csv')

    def statistic_non_feature_num(self):
        day = 8
        df = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        df_train = df[df['pt_d'] < day]
        df_test = df[df['pt_d'] == day]
        tmp = []
        for feature_name in sparse_feature:
            train_set = set(df_train[feature_name])
            test_set = set(df_test[feature_name])
            test_diff = test_set - train_set
            tmp.append((len(test_diff), len(test_set), len(train_set), len(test_diff)/len(train_set)))
        pd.DataFrame(tmp, index=sparse_feature, columns=['diff', 'test_set', 'train_set', 'test_diff_train_ratio']).\
            to_csv('./yangzhe/statistic/feature_distribution_{}.csv'.format(day))

    def get_test_uid_distribution(self):
        # 这个是已经排序好的
        data = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        df_test = data[data.pt_d == 8]
        df_train = data[data.pt_d <= 7]
        tmp_df = df_train.groupby('uid', as_index=False, sort=False)['pt_d'].last()
        df_test = pd.merge(df_test[['uid']], tmp_df, on='uid', how='left')
        print(df_test['pt_d'].value_counts(dropna=False))

    def get_click_distribution(self):
        # 统计下曝光一次即点击一次的用户所点击的广告的全局点击次数(流行程度)
        data = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
        data = data[data.pt_d <= 7]
        tmp = data['uid']
        counts = tmp.value_counts()
        no_need_id = counts[counts <= 1].index
        no_need_index = tmp[tmp.isin(no_need_id)].index
        filter_data = data.loc[no_need_index]
        print('只有一次曝光的用户为:', filter_data.shape[0])
        filter_data = filter_data[filter_data.label == 1]
        print('只有一次曝光且点击的用户为:', filter_data.shape[0])

        # 看一下这些用户都倾向于点击哪个task_id,以及这些task_id的全局点击率
        task_id = filter_data['task_id']
        task_id.value_counts().to_csv('./yangzhe/statistic/task_id_1_click_distribution.csv')
        task_id_click = data.loc[data.label == 1, 'task_id']
        task_id_click = task_id_click[task_id_click.isin(task_id)]
        task_id_click.value_counts().to_csv('./yangzhe/statistic/task_id_1_click_all_distribution.csv')
        # 发现仅有几个这些用户点击的task_id大部分都是热门task_id,也就是说新用户/不活跃用户往往喜欢点击较为热门的商品

    # 统计缺失率最高的前50个特征，删除
    def get_top50_non_feature(self):
        df = pd.read_csv('./yangzhe/non/2th_non_value_df.csv', index_col=0)
        for day in range(3, 9):
            df += pd.read_csv('./yangzhe/non/{}th_non_value_df.csv'.format(day), index_col=0)
        df['non_rate'] = df['non_rate'] / 7
        df = df.sort_values(by='non_rate', ascending=False)
        print(df[df.non_rate > 0.1].index)

    def get_feature_importance(self):
        df_list = []
        for mode in ['lightgbm_2_3', 'lightgbm_4_5', 'lightgbm_7']:
            for flod in range(0, 5):
                file_path = './yangzhe/feature_importance/{}_{}.csv'.format(mode, flod)
                df_list.append(pd.read_csv(file_path, index_col=0))
        df = pd.concat(df_list, axis=1, ignore_index=True, sort=False)
        df['importance_mean'] = df.mean(axis=1)
        df = df.sort_values(by='importance_mean', ascending=False)
        index_list = [x for x in df[df.importance_mean > 50].index if x not in sparse_feature]
        joblib.dump(index_list, './yangzhe/feature_importance/importance_index.pkl')


if __name__ == "__main__":
    statistic = Statistic()
    # statistic.get_user_mean_click()
    # statistic.get_ctr_per_day()
    # statistic.get_feature_importance()
    statistic.statistic_non_feature_num()
    # statistic.get_top50_non_feature()


