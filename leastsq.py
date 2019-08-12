import scipy as sp
import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import os
import chess

# TODO: Do we need a bit for bias?

tensor = False
# 0 for no, 1 for standard, 2 for extended, 3 for fake
tensor_sketch = 0
add_counts = False
print_table = True
# One of svd, lsmr, lsqr, lstsq
# lstsq is needed for tensor_sketch
method = 'lsmr'
input_file = 'games.out'
res_file = 'games.out.res'

# Would be nice to try l1 regression
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_dense_vs_sparse_data.html

print('Loading data')
X = sp.sparse.load_npz(input_file + '.npz')
X = sp.sparse.csr_matrix(X, dtype=float)
Y = np.load(res_file + '.npy')

if tensor:
    print('tensoring')
    tens_name = input_file + '.tens.npz'
    if os.path.isfile(tens_name):
        X = sp.sparse.load_npz(tens_name)
    else:
        X = sp.sparse.vstack((row.T@row).reshape(1, -1) for row in X).tocsr()
        sp.sparse.save_npz(tens_name, X)
    print('tensoring done')

if tensor_sketch != 0:
    if method != 'lstsq':
        print('Tensor sketch makes dense matrices, so we have to use method=lstsq')
    print(f'Doing the sketch {tensor_sketch}')
    n, d = X.shape
    m = d  # make it larger?
    if tensor_sketch == 1:
        M = np.random.randint(-1, 2, (d, m)) / (m * 2 / 3)**.5
        T = np.random.randint(-1, 2, (d, m)) / (m * 2 / 3)**.5
        X = (X@M) * (X@T)
    elif tensor_sketch == 2:
        M = np.random.randint(-1, 2, (d**2, m)) / (m * 2 / 3)**.5

        def blocks():
            bs = 10000
            for i in range(0, n, bs):
                Y = X[i:i + bs].todense()
                Y = sp.einsum('ij,ik->ijk', Y, Y).reshape(Y.shape[0], -1)
                Y = Y@M
                yield Y
        X = np.vstack(blocks())
    elif tensor_sketch == 3:
        # No idea why you'd want to do this, but seems to work
        M = np.random.randint(-1, 2, (d, m)) / (m * 2 / 3)**.5
        X = X@M

if add_counts:
    print('Adding counts')
    cols = []
    for i in range(2 * 6):
        cols.append(X[:, i * 64:(i + 1) * 64].sum(axis=1))
    X = sp.sparse.hstack([X] + cols).tocsr()

n, d = X.shape
split = .9
ntrain = int(n * split)
Xtrain, Xtest = X[:ntrain], X[ntrain:]
Ytrain, Ytest = Y[:ntrain], Y[ntrain:]


def print_tables(w):
    for i, color in enumerate([chess.WHITE, chess.BLACK]):
        for j, ptype in enumerate(range(chess.PAWN, chess.KING + 1)):
            table = w[j * 64 + i * 64 * 6:(j + 1) * 64 + i * 64 * 6].reshape(8, 8)
            print(chess.Piece(ptype, color))
            if add_counts:
                print('Val:', w[12 * 64 + 6 * i + j])
            print(table.round(2))


print('Running regressions')
for rows in [ntrain // 30, ntrain // 10, ntrain // 3, ntrain]:
    Xr = Xtrain[:rows]
    Yr = Ytrain[:rows]
    # Perhaps we never need to consider r > d-32, since we know we have that rank
    if method == 'svd':
        for rank in [2**i for i in range(6, 10) if 2**i <= d] + [d - 40, d - 32]:
            if rank < d:
                # Should use SM (smallest) rather than LM, but it appears broken
                U, S, V = sp.sparse.linalg.svds(Xr, k=rank, which='LM')
            else:
                U, S, V = np.linalg.svd(Xr.todense(), full_matrices=False)
                # Something weird happens with my shapes if I don't do this
                U, V = np.array(U), np.array(V)
            w = V.T @ (np.linalg.pinv(np.diag(S)) @ (U.T @ Yr))
            loss = np.sum((Xtest @ w - Ytest)**2) / len(Ytest)
            print(f'Rank: {rank}, n: {rows}, Loss: {loss}')
        print()
    else:
        if method == 'lstsq':
            w = sp.linalg.lstsq(Xr, Yr)[0]
        elif method == 'lsqr':
            w = sp.sparse.linalg.lsqr(Xr, Yr)[0]
        elif method == 'lsmr':
            w = sp.sparse.linalg.lsmr(Xr, Yr)[0]
        loss = np.sum((Xtest @ w - Ytest)**2) / len(Ytest)
        print(f'n: {rows}, Loss: {loss}')

if print_table:
    print(f'Full train ({n})')
    w = sp.sparse.linalg.lsmr(X, Y)[0]
    print_tables(w)
