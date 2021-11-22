import numpy as np
import cugraph
import cudf
import dgl
from dgl.sampling import random_walk, pack_traces
import torch
import scipy.sparse
import time
import cupy as cp
import random
import math
import os
import torch as th
import dgl.function as fn
import random


class CugraphSampler(object):
    def __init__(self, dn, g, train_nid, node_budget, num_roots, length, num_repeat=50):
        """
        :param dn: name of dataset
        :param g: full graph
        :param train_nid: ids of training nodes
        :param node_budget: expected number of sampled nodes
        :param num_repeat: number of times of repeating sampling one node
        """
        self.g = g
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_repeat = dn, num_repeat
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None

        self.num_roots = num_roots
        self.length = length

        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0

            t = time.perf_counter()
            while sampled_nodes <= self.train_g.num_nodes() * num_repeat:
                subgraph = self.__sample__()
                self.subgraphs.append(subgraph)
                sampled_nodes += subgraph.shape[0]
                self.N += 1
            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            # np.save(graph_fn, self.subgraphs)

            t = time.perf_counter()
            self.__counter__()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            # np.save(norm_fn, (aggr_norm, loss_norm))

        self.train_g.ndata['l_n'] = th.Tensor(loss_norm)
        self.train_g.edata['w'] = th.Tensor(aggr_norm)
        self.__compute_degree_norm()

        self.num_batch = math.ceil(self.train_g.num_nodes() / node_budget)
        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))
        print("The size of subgraphs is about: ", len(cp.unique(self.subgraphs[-1])))

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __counter__(self):

        for sampled_nodes in self.subgraphs:

            sampled_nodes_df = cudf.DataFrame(sampled_nodes)

            sampled_nodes = th.from_numpy(sampled_nodes).long()
            self.node_counter[sampled_nodes] += 1


            edge_list= cudf.DataFrame()
            edge_list['src'] = sampled_nodes_df.drop(sampled_nodes_df.iloc[[(self.length + 1)*i + self.length for i in range(self.num_roots)]].index).reset_index(drop=True)
            edge_list['dst'] = sampled_nodes_df.drop(sampled_nodes_df.iloc[[(self.length + 1)*i for i in range(self.num_roots)]].index).reset_index(drop=True)
            src_ids = th.tensor(edge_list['src'])
            dst_ids = th.tensor(edge_list['dst'])
            subg = dgl.graph((src_ids, dst_ids),idtype=th.int64)

            # subg = self.train_g.subgraph(sampled_nodes)
            # sampled_edges = subg.edata[dgl.EID]
            sampled_edges = subg.edge_ids(src_ids.long(), dst_ids.long())
            self.edge_counter[sampled_edges] += 1

    def __generate_fn__(self):
        raise NotImplementedError

    def __compute_norm__(self):
        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batch:
            result = self.train_g.subgraph(self.subgraphs[self.n])
            self.n += 1
            return result
        else:
            random.shuffle(self.subgraphs)
            raise StopIteration()


# """"
# Define random walk sampler using cuGraph random walk API
# """"
class CugraphRWSampler(CugraphSampler):
    def __init__(self, num_roots, length, dn, g, _g, train_nid, num_repeat=50):
        self._g = _g
        # self.num_roots, self.length = num_roots, length
        super(CugraphRWSampler, self).__init__(dn, g, train_nid, num_roots * length, num_roots, length, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):

        _g_nodes = self._g.nodes().to_array().tolist()
        sampled_roots = random.sample(_g_nodes, self.num_roots)

        rw_path, _, _= cugraph.random_walks(self._g, sampled_roots, self.length+1)

        sampled_nodes = cp.asnumpy(rw_path)
        del rw_path
        return sampled_nodes
