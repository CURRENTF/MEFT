import faiss
import hnswlib
import numpy as np
import pickle
import torch
import timeit
from faiss_vector_database import *
from sklearn.cluster import KMeans
from collections import defaultdict


class HNSW:
    def __init__(self, key_num, key_dim, data, ef_s=None, ef_c=200, M=16, topk=16, num_threads=16):
        self.p = hnswlib.Index(space='ip', dim=key_dim)
        self.p.set_num_threads(num_threads)
        self.p.init_index(max_elements=key_num, ef_construction=ef_c, M=M, random_seed=42)
        self.p.add_items(data, np.arange(key_num))
        self.topk = topk
        if ef_s is None:
            self.p.set_ef(topk + 1)
        else:
            self.p.set_ef(ef_s)

    def query(self, q):
        labels, distances = self.p.knn_query(q, k=self.topk)
        return labels, distances


class FlatTest:
    def __init__(self, data, topk=16):
        self.vecs = data
        self.topk = topk

    def query(self, q):
        _ = np.matmul(self.vecs, q.T) * -1
        __ = np.argsort(_, axis=0)
        vals = np.sort(_, axis=0)
        return __[:self.topk, :], vals[:self.topk, :] * -1


class KmeansIndex:
    def __init__(self, X, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.index = defaultdict(list)
        self.fit(X)

    def fit(self, X):
        # 使用k-means算法拟合数据
        self.X = X
        labels = self.kmeans.fit_predict(X)
        # 创建倒排索引
        for i, label in enumerate(labels):
            self.index[label].append(i)

    def search(self, vectors, topk=64):
        # 预测给定向量的聚类
        labels = self.kmeans.predict(vectors)
        # 计算内积并返回内积最高的k个向量
        result = []
        for i, label in enumerate(labels):
            indices = self.index[label]
            scores = torch.matmul(torch.tensor(self.X[indices]), torch.tensor(vectors[i]))
            topk_indices = torch.topk(scores, k=topk).indices
            result.append([indices[i] for i in topk_indices])
        return np.array(result), None


def test_hnsw(dim, m, ef_s, ef_c, q_num=32, k_num=11008, topk=16):
    query = torch.randn((q_num, dim)).numpy()
    keys = torch.randn((k_num, dim)).numpy()
    print("start build")
    index = HNSW(key_num=k_num, key_dim=dim, data=keys, ef_s=ef_s, ef_c=ef_c, M=m, topk=topk, num_threads=32)
    print("build done")
    flat_index = FlatTest(data=keys, topk=topk)

    def test():
        res = index.query(query)
        return res

    def compute_recall():
        res_hnsw = index.query(query)
        res_flat = flat_index.query(query)
        # print(res_flat[0].shape)
        # print(res_flat[0])
        # print("---- split ----")
        # print(res_flat[1])
        # print("---- split ----")
        # res_flat = flat_index.search(query, k=topk)
        print("res_hnsw\n", res_hnsw, file=ooo)
        print("res_flat\n", res_flat, file=ooo)
        print("shape compare", res_hnsw[0].shape, res_flat[0].shape)
        overlap = sum([len(set(a).intersection(b)) for a, b in zip(res_hnsw[0], res_flat[0].T)])
        total = res_hnsw[0].size
        recall = overlap / total
        print(f"overlap: {overlap}, total: {total}")
        return recall

    times = 4
    test_res = timeit.timeit(test, number=times)
    recall = compute_recall()
    print(f"dim={dim}, m={m}, ef_s={ef_s}, ef_c={ef_c}, topk={topk}, k_num={k_num}, dim={dim}")
    print("final test res", test_res / times, 's')
    print("recall", recall)
    print(f"{'-' * 40} test end {'-' * 40}")


def get_IVFxPQy(xb, dim_k):
    dim, measure = dim_k, faiss.METRIC_INNER_PRODUCT
    param = 'IVF128,Flat'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为False，因为倒排索引需要训练k-means，
    index.train(xb)  # 因此需要先训练index，再add向量 index.add(xb)
    index.add(xb)
    return index


def create_faiss_index(data, dimension):
    """
    创建一个faiss倒排索引

    参数:
    data (numpy array): 一个二维numpy数组，每一行是一个向量
    dimension (int): 向量的维度

    返回:
    index (faiss index): 一个faiss倒排索引
    """
    assert data.shape[1] == dimension

    # 初始化索引
    quantizer = faiss.IndexFlatIP(dimension)  # the other index
    index = faiss.IndexIVFFlat(quantizer, dimension, 128)

    assert not index.is_trained
    index.train(data)
    assert index.is_trained

    # 添加数据到索引
    index.add(data)
    index.nprobe = 64
    return index


def create_faiss_flat_index(data, dimension):
    """
    创建一个faiss flat索引

    参数:
    data (numpy array): 一个二维numpy数组，每一行是一个向量
    dimension (int): 向量的维度

    返回:
    index (faiss index): 一个faiss flat索引
    """
    assert data.shape[1] == dimension

    # 初始化索引
    index = faiss.IndexFlatIP(dimension)

    # 添加数据到索引
    index.add(data)

    return index


def test_IVFxPQy(dim, q_num=8, k_num=11008, topk=4):
    query = torch.randn((q_num, dim)).numpy()
    keys = torch.randn((k_num, dim)).numpy()
    # index = get_IVFxPQy(keys, dim)
    index = create_faiss_index(keys, dim)
    # index = create_faiss_flat_index(keys, dim)
    flat_index = FlatTest(data=keys, topk=topk)

    print("build done")

    def test():
        res = index.search(query, k=topk)
        return res

    def flat_test():
        res = flat_index.query(query)
        return res

    def compute_recall():
        res_hnsw = index.search(query, k=topk)
        res_flat = flat_index.query(query)
        print(res_hnsw[1].shape, res_flat[0].shape)
        print("res_hnsw_dis\n", res_hnsw[0])
        print("res_hnsw_index\n", res_hnsw[1])
        print("res_flat_dis\n", res_flat[1].T)
        print("res_flat_index\n", res_flat[0].T)
        overlap = 0
        for a, b in zip(res_hnsw[1], res_flat[0].T):
            if len(a) != len(b):
                raise ValueError
            overlap += len(set(a).intersection(b))
        total = res_hnsw[0].size
        recall = overlap / total
        print(f"overlap: {overlap}, total: {total}")
        return recall

    times = 10
    test_res = timeit.timeit(test, number=times)
    recall = compute_recall()
    print("final test res", test_res / times, 's')
    print("recall", recall)
    print(f"{'-' * 40} test end {'-' * 40}")

    times = 10
    test_res = timeit.timeit(flat_test, number=times)
    print("flat test res", test_res / times, 's')


def test_KmeansIndex(dim, q_num=8, k_num=11008, topk=64):
    query = torch.randn((q_num, dim)).numpy()
    keys = torch.randn((k_num, dim)).numpy()
    index = KmeansIndex(keys, 256)
    flat_index = FlatTest(data=keys, topk=topk)

    print("build done")

    def test():
        res = index.search(query, topk)
        return res

    def flat_test():
        res = flat_index.query(query)
        return res

    def compute_recall():
        res_hnsw = index.search(query, topk)
        res_flat = flat_index.query(query)
        print(res_hnsw[0].shape, res_flat[0].shape)
        print("res_hnsw_dis\n", res_hnsw[0])
        print("res_hnsw_index\n", res_hnsw[1])
        print("res_flat_dis\n", res_flat[1].T)
        print("res_flat_index\n", res_flat[0].T)
        overlap = 0
        for a, b in zip(res_hnsw[0], res_flat[0].T):
            if len(a) != len(b):
                raise ValueError
            overlap += len(set(a).intersection(b))
        total = res_hnsw[0].size
        recall = overlap / total
        print(f"overlap: {overlap}, total: {total}")
        return recall

    times = 10
    test_res = timeit.timeit(test, number=times)
    recall = compute_recall()
    print("final test res", test_res / times, 's')
    print("recall", recall)
    print(f"{'-' * 40} test end {'-' * 40}")

    times = 10
    test_res = timeit.timeit(flat_test, number=times)
    print("flat test res", test_res / times, 's')


if __name__ == "__main__":
    torch.manual_seed(42)
    ooo = open("debug.txt", 'w', encoding='utf-8')
    ms = [128]
    ef_ss = [128]
    ef_cs = [512, 2048, 4096]
    k_nums = [10000, 50000, 100000]
    dims = [512, 4096]
    for m in ms:
        for ef_s in ef_ss:
            for ef_c in ef_cs:
                for k_num in k_nums:
                    for dim in dims:
                        test_hnsw(dim, m, ef_s, ef_c, k_num=k_num)

    # test_KmeansIndex(4096)
    ooo.close()
