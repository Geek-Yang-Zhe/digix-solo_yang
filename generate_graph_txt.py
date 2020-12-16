import pandas as pd
import os
import numpy as np
import networkx as nx
import pickle
from gensim.models import Word2Vec
import joblib
from joblib import Parallel, delayed
import itertools


def load_training_data(f_name):
    # 将训练集中的边存储为元组形式，返回{edge_type : [(node1, node2)]}
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def df2txt():
    all_feature = pd.read_pickle('./yangzhe/feature/all/all_feature.pkl')
    all_feature.loc[all_feature['pt_d'] == 8, 'label'] = 0
    need_df = all_feature.loc[(all_feature['label'] == 1) & (all_feature['pt_d'] <= 3), ['uid', 'task_id']]
    need_df['edge_type'] = 'click'
    need_df.to_csv('./yangzhe/graph/1_3_click.txt', index=False, columns=['edge_type', 'uid', 'task_id'], header=None,
                   sep=' ')
    # 存储节点类别
    node_type_df = pd.DataFrame()
    unique_uid, unique_task_id = need_df['uid'].unique(), need_df['task_id'].unique()
    node_type_df['node'] = np.concatenate((unique_uid, unique_task_id), axis=0)
    node_type_df['node_type'] = ['U'] * len(unique_uid) + ['I'] * len(unique_task_id)
    node_type_df.to_csv('./yangzhe/graph/1_3node_type.txt', index=False, columns=['node', 'node_type'], header=None,
                        sep=' ')


def construct_graph(edge_list):
    G = nx.Graph()
    for u, v in edge_list:
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v)
            G[u][v]['weight'] = 1
    return G


def generate_walks(edge_dict, num_walks=20, walk_length=10, all_schema='U-I-U,I-U-I'):
    print('当前num_walks为: {}'.format(num_walks))
    node_type_dict = load_node_type('./yangzhe/graph/1_3node_type.txt')
    for edge_type in edge_dict:
        graph = construct_graph(edge_dict[edge_type])
        meta_walk = MetaWalk(graph=graph, node_type_dict=node_type_dict)
        walks = meta_walk.simulate_walks(num_walks=num_walks, walk_length=walk_length, all_schema=all_schema)
        # 存储walks
        print(walks)
        joblib.dump(walks, './yangzhe/graph/{}_{}walks_{}length.pkl'.format(edge_type, num_walks, walk_length))


class MetaWalk:
    def __init__(self, graph, node_type_dict):
        self.G = graph
        self.node_type_dict = node_type_dict

    def walk(self, node, walk_length, schema):
        schema_seq = schema.split('-')
        walk_seq = [node]
        while len(walk_seq) < walk_length:
            cur = walk_seq[-1]
            schema_idx = len(walk_seq) % (len(schema_seq) - 1)
            need_neighbors = [x for x in self.G[cur].keys() if self.node_type_dict[x] == schema_seq[schema_idx]]
            if need_neighbors:
                walk_seq.append(np.random.choice(need_neighbors))
            else:
                break
        return walk_seq

    def meta_simulate_walks(self, nodes, walk_length, all_schema):
        schema_list = all_schema.split(',')
        walks = []
        # 游走次数，每个节点游走长度，起始游走点和和当前schema一致。
        for i, node in enumerate(nodes):
            for schema in schema_list:
                if self.node_type_dict[node] == schema.split('-')[0]:
                    walks.append(self.walk(node, walk_length, schema))
            if i % 200000 == 0:
                print('迭代次数为{}次'.format(i))
        return walks

    def simulate_walks(self, num_walks, walk_length, all_schema, n_jobs=10):
        G = self.G
        nodes = list(G.nodes())
        walks = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self.meta_simulate_walks)(nodes, walk_length, all_schema) for _ in range(num_walks))
        walks = list(itertools.chain(*walks))
        return walks

    def no_parallel_walks(self, num_walks, walk_length, all_schema):
        G = self.G
        nodes = list(G.nodes())
        walks = self.meta_simulate_walks(nodes, walk_length, all_schema)
        return walks


def word2vec(edge_type, num_walks, walk_length):
    with open('./yangzhe/graph/{}_{}walks_{}length.pkl'.format(edge_type, num_walks, walk_length), 'rb') as f:
        walks = pickle.load(f)
    model = Word2Vec(sentences=walks, min_count=0, size=16, sg=1, negative=5, window=5, iter=5, workers=19)
    model.save('./yangzhe/graph/{}_{}emb_{}iter_{}window.model'.format(edge_type, 16, 5, 5))


if __name__ == '__main__':
    if not os.path.exists('./yangzhe/graph/1_3_click.txt'):
        df2txt()
    edge_data_by_type = load_training_data('./yangzhe/graph/1_3_click.txt')
    generate_walks(edge_data_by_type, num_walks=10, walk_length=10)
    word2vec(edge_type='click', num_walks=10, walk_length=10)
