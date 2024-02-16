import faiss
import torch
import timeit

faiss.omp_set_num_threads(12)


def get_phi(xb):
    return (xb ** 2).sum(1).max()


def augment_xb(xb, phi=None):
    norms = (xb ** 2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = torch.sqrt(phi - norms)
    return torch.cat((xb, extracol.view(-1, 1)), dim=1)


def augment_xq(xq):
    extracol = torch.zeros(len(xq))
    return torch.cat((xq, extracol.view(-1, 1)), dim=1)


class HNSWIndex:
    def __init__(self, dim=4096, m=64, ef_search=32, ef_construction=64, x=None):
        self.index = faiss.IndexHNSWFlat(dim + 1, m)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        if x is not None and isinstance(x, torch.Tensor):
            self.add_vectors(vectors=x)

    def add_vectors(self, vectors: torch.Tensor):
        print("building index ... ")
        n = vectors.shape[0]
        vectors = augment_xb(vectors)
        vectors = vectors.numpy()
        self.index.add(vectors)
        print("build done")

    def topk(self, query: torch.Tensor, topk=16):
        query = augment_xq(query)
        query = query.numpy()
        return self.index.search(query, k=topk)


def performance_test(dim, m, ef_s, ef_c):
    xq = torch.randn((64, dim))
    xb = torch.randn((11008, dim))
    index = HNSWIndex(dim, m, ef_s, ef_c)
    index.add_vectors(xb)
    def test():
        res = index.topk(xq)
        return res

    times = 100
    test_res = timeit.timeit(test, number=times)
    print(f"dim={dim}, m={m}, ef_s={ef_s}, ef_c={ef_c}")
    print("final test res", test_res / times, 's')


def performance_test_(dim, m, ef_s, ef_c, q_num=64, k_num=11008):
    xq = torch.randn((q_num, dim))
    xb = torch.randn((k_num, dim))
    index = HNSWIndex(dim, m, ef_s, ef_c)
    index.add_vectors(xb)

    # create a flat index as ground truth
    flat_index = faiss.IndexFlatL2(dim + 1)
    flat_index.add(augment_xb(xb).numpy())

    def test():
        res = index.topk(xq)
        return res

    def compute_recall():
        res_hnsw = index.topk(xq)
        res_flat = flat_index.search(augment_xq(xq).numpy(), k=16)
        overlap = sum([len(set(a).intersection(b)) for a, b in zip(res_hnsw[1], res_flat[1])])
        total = res_hnsw[0].size
        recall = overlap / total
        return recall

    times = 10
    test_res = timeit.timeit(test, number=times)
    recall = compute_recall()
    print(f"dim={dim}, m={m}, ef_s={ef_s}, ef_c={ef_c}")
    print("final test res", test_res / times, 's')
    print("recall", recall)


if __name__ == "__main__":
    ms = [8, 16, 32, 64]
    ef_ss = [8, 16, 32, 64, 128]
    ef_cs = [8, 16, 32, 64]
    for m in ms:
        for ef_s in ef_ss:
            for ef_c in ef_cs:
                performance_test_(4096, m, ef_s, ef_c)
