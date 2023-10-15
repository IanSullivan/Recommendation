import numpy as np


def generate_random_vectors(dim, n_vectors):
    return np.random.randn(dim, n_vectors)


def build_lsh(data, n_vectors):
    dim = data.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)

    # Partition data points into bins,
    bin_indices_bits = data.dot(random_vectors) >= 0
    # and encode bin index bits into integers
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)
    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = dict()
    for idx, bin_index in enumerate(bin_indices):
        pass
        # if bin_index not in table:
        #     table[bin_index] = []

        # Fetch the list of document ids associated with the bin and add the document id to the end.
        # table[bin_index].append(idx)  # YOUR CODE HERE

    # Note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'data': data,
             'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model


rng = np.random.default_rng(12345)
data = []
arr1 = rng.random((21, 3, 3))
# for i in range(20):
#     data.append(rng.integers(low=0, high=10, size=3))
# print(arr1)
build_lsh(arr1, 3)
