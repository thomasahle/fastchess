import scipy as sp
import sys
import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import os
import chess
import argparse
import sklearn.linear_model
import time
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('-xin', help='input, .npz')
parser.add_argument('-yin', help='labels, .npy')
parser.add_argument('-cout', help='coefficients out')
parser.add_argument(
    '-solver', help='‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’')
parser.add_argument('-penalty', default='l2',
                    help='l1, l2, elasticnet or none')
parser.add_argument('-maxiter', default=100)
parser.add_argument('-verbose', action='store_true')
parser.add_argument('-multi', default='multinomial',
                    help='ovr or multinomial or crammer_singer (svc)')
parser.add_argument('-model', help='One of logistic, svc or sgd')
parser.add_argument('-loss', default='squared_hinge',
                    help='(for svc) hinge or squared_hinge')
parser.add_argument('-nodual', action='store_true',
                    help='(for svc) useful if n >> d')
args = parser.parse_args()
print(args)

# best appears to be solver=sag, penalty=none

print('Loading data')
X = sp.sparse.load_npz(args.xin)
X = sp.sparse.csr_matrix(X)
Y = np.load(args.yin)

n, d = X.shape
split = .5
ntrain = int(n * split)
Xtrain, Xtest = X[:ntrain], X[ntrain:]
Ytrain, Ytest = Y[:ntrain], Y[ntrain:]
# Moves actually ever played
classes = np.unique(Y)

print('Shuffling')
shuffle_index = np.arange(ntrain)
np.random.shuffle(shuffle_index)
Xtrain = Xtrain[shuffle_index, :]
Ytrain = Ytrain[shuffle_index]

if args.model == 'svc':
    clf = sklearn.svm.LinearSVC(verbose=args.verbose, multi_class=args.multi,
                                max_iter=args.maxiter, penalty=args.penalty, loss=args.loss, dual=not args.nodual)
    # Prefer dual=False when n_samples > n_features
elif args.model == 'logistic':
    clf = sklearn.linear_model.LogisticRegression(
        solver=args.solver, verbose=args.verbose, multi_class=args.multi, max_iter=args.maxiter, penalty=args.penalty)
elif args.model == 'sgd':
    clf = sklearn.linear_model.SGDClassifier(
        verbose=args.verbose, max_iter=args.maxiter, penalty=args.penalty)
else:
    print('please choose a model')
    sys.exit()

print('Making batches')
# for rows in [ntrain//30, ntrain//10, ntrain//3, ntrain]:
if args.model == 'sgd':
    bs = 100
    batches = ((i, i+bs) for i in range(0, ntrain, bs))
else:
    batches = [(0, 2**i)
               for i in range(5, 30) if 2**i < ntrain] + [(0, ntrain)]

print('Running regressions')
# with joblib.parallel_backend('threading'):
for r1, r2 in batches:
    Xr = Xtrain[r1:r2]
    Yr = Ytrain[r1:r2]
    t = time.time()
    if args.model == 'sgd':
        clf.partial_fit(Xr, Yr, classes=classes)
    else:
        clf.fit(Xr, Yr)
    t = time.time() - t
    acc = np.sum(clf.predict(Xtest) == Ytest) / (n-ntrain)
    # TODO: amount of valid moves
    rows = r2-r1
    print(
        f'Rows: {rows}, Accuracy: {acc:.3%}, Time: {t:.1f}s, Per row/s: {rows/t:.1f}')
    #np.save(args.cout, np.hstack([clf.coef_, clf.intercept_[:,None]]))
    joblib.dump(clf, args.cout + '.joblib')
