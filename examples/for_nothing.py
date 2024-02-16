# python for_nothing.py

def test1():
    import time

    import numpy as np
    import faiss

    d = 2048  # dimension
    nb = 4096  # database size
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')

    index = faiss.IndexFlatIP(d)
    # index = faiss.IndexIDMap(index)
    # index.add_with_ids(xb, np.arange(nb))
    index.add(xb)
    s = time.time()
    index.remove_ids(np.arange(2600, 4000))
    e = time.time()
    print('time used:', e - s)
    print(index.is_trained)
    print(index.ntotal)

    for i in range(1):
        k = 128
        nq = 1  # number of queries
        xq = np.random.random((nq, d)).astype('float32')
        s = time.time()
        D, I = index.search(xq, k)  # sanity check
        e = time.time()
        print('time used:', e - s)
        print(I[0])
        # print(I[:5])  # neighbors of the 5 first queries
        # print(D[:5])  # distances of the 5 first queries


def test2():
    from datasets import load_dataset
    data = load_dataset("json", data_files='./dataset/gsm8k/my_train.json')
    print(type(data))
    print(len(data['train']))
    train_val = data["train"].train_test_split(
        test_size=10, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle()
    )
    val_data = (
        train_val["test"].shuffle()
    )
    print(val_data)
    print(len(train_data))
    from datasets.formatting import formatting


test2()