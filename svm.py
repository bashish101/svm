import cvxopt
import itertools
import numpy as np
import multiprocessing as mp

from sklearn import svm, datasets

cvxopt.solvers.options['show_progress'] = False
KERNELS = ['linear', 'poly', 'rbf']

class SVMUnit(object):
	def __init__(self,
		     classes,
		     kernel = 'linear',
		     sigma = 5.0,
		     C = None,
		     strategy = 'one_vs_one'):
		if type(kernel) == str:
			if kernel not in KERNELS:
				print ('Unknown kernel type. Switching to linear kernel.')
				self.kernel = self.linear
			elif kernel == 'poly':
				self.kernel = self.polynomial
			elif kernel == 'rbf':
				self.kernel = self.gaussian
			elif kernel == 'linear':
				self.kernel = self.linear
		elif hasattr(kernel, '__call__'):
			self.kernel = kernel
		else:
			print ('Unknown kernel. Switching to linear kernel.')
			self.kernel = self.linear_kernel
		
		self.sigma = sigma
		self.C = C

		self.strategy = strategy
		self.val_to_cls = {1 : classes[0], -1 : classes[1]}
		self.cls_to_val = {**{cls : 1 for cls in classes[0]}, **{cls : -1 for cls in classes[1]}}

	def val_to_cls_lookup(self, y):
		if self.strategy == 'one_vs_one':
			return np.vectorize(lambda val:self.val_to_cls[int(val)][0])(y)
		elif self.strategy == 'one_vs_rest':
			return np.vectorize(lambda val:self.val_to_cls[1][0])(y)

	def cls_to_val_lookup(self, y):
		return np.vectorize(lambda cls:self.cls_to_val[cls])(y)

	def linear(self, x1, x2):
		return np.dot(x1, x2)

	def polynomial(self, x1, x2, p = 3):
		return (1 + np.dot(x1, x2)) ** p

	def gaussian(self, x1, x2):
		return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.sigma ** 2)))

	def fit(self, X, y):
		size, num_feats = X.shape
		y = self.cls_to_val_lookup(y)

		K = np.zeros((size, size))
		for idx1 in range(size):
			for idx2 in range(size):
				K[idx1, idx2] = self.kernel(X[idx1], X[idx2])

		P = cvxopt.matrix(np.outer(y, y) * K, tc = 'd')
		q = cvxopt.matrix(np.ones(size) * -1)
		A = cvxopt.matrix(y, (1, size), tc = 'd')
		b = cvxopt.matrix(0.0)

		if self.C is None:
			G = cvxopt.matrix(np.diag(np.ones(size) * -1))
			h = cvxopt.matrix(np.zeros(size))
		else:
			tmp_mat1 = np.identity(size) * -1
			tmp_mat2 = np.identity(size)
			G = cvxopt.matrix(np.vstack((tmp_mat1, tmp_mat2)))
			lower_bound = np.zeros(size)
			upper_bound = np.ones(size) * self.C
			h = cvxopt.matrix(np.hstack((lower_bound, upper_bound)))

		solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		lagr_mult = np.ravel(solution['x'])

		selector = lagr_mult > 1e-5
	
		idx_list = np.arange(len(lagr_mult))[selector]

		self.alphas = lagr_mult[selector]
		self.support_vectors = X[selector].astype(int)
		self.support_vectors_labels = y[selector].astype(int)
		
		self.b = 0
		for idx in range(len(self.alphas)):
			self.b += self.support_vectors_labels[idx]
			self.b -= np.sum(self.alphas * self.support_vectors_labels * K[idx_list[idx], selector])
		self.b /= len(self.alphas)

		if self.kernel == self.linear:
			self.w = np.zeros(num_feats)
			for idx in range(len(self.alphas)):
				self.w += self.alphas[idx] * self.support_vectors_labels[idx] * self.support_vectors[idx]
		else:
			self.w = None

	def project(self, X):
		if self.kernel == self.linear:
			y = np.dot(X, self.w)
		else:
			y = []
			for idx in range(len(X)):
				s = 0
				sv_zip = zip(self.alphas, self.support_vectors_labels, self.support_vectors)
				for alpha, support_vector_label, support_vector in sv_zip:
					s += alpha * support_vector_label * self.kernel(X[idx], support_vector)

				y.append(s)
			y = np.array(y)
		y += self.b
		return y

	def predict(self, X):
		if self.strategy == 'one_vs_one':
			return np.sign(self.project(X))
		else:
			return self.project(X)

class SVM(object):
	def __init__(self, 
		     kernel = 'linear',
		     sigma = 5.0,
		     C = None,
		     classes = None,
		     strategy = 'one_vs_one'):
		self.kernel_type = kernel
		
		self.sigma = sigma
		self.C = C
		self.classes = classes
		self.strategy = strategy

		if self.strategy == 'one_vs_one':
			self.pairs = self.one_vs_one_pairs
			self.predict = self.one_vs_one_predict
		else:
			self.pairs = self.one_vs_rest_pairs
			self.predict =  self.one_vs_rest_predict

		self.svm_unit_list = []

	def one_vs_one_pairs(self, classes):
		return [([pair[0]], [pair[1]]) for pair in list(itertools.combinations(classes, 2))]

	def one_vs_rest_pairs(self, classes):
		classes = list(classes)
		return [([classes[idx]], list(classes[:idx]) + list(classes[idx + 1:])) for idx in range(len(classes))]

	def create_svm_unit(self, data):
		X, y, classes = data
		
		unit = SVMUnit(classes = classes,
			       kernel = self.kernel_type,
			       sigma = self.sigma,
			       C = self.C,
			       strategy = self.strategy)
		unit.fit(X, y)

		return unit

	def generate_subsets(self, pairs, data):
		data_subsets = []
		for pair in pairs:
			feat_list = []
			lbl_list = []
			filter_classes = [cls for elm in pair for cls in elm]
			for feat, lbl in data:
				if lbl in filter_classes:
					feat_list.append(feat)
					lbl_list.append(lbl)
			feat_list = np.array(feat_list, dtype = np.float64)
			lbl_list = np.array(lbl_list, dtype = np.float64)
			data_subsets.append((feat_list, lbl_list, pair))

		return data_subsets

	def fit(self, X, y):
		if self.classes is None:
			self.classes = np.unique(y)
	
		pairs = self.pairs(self.classes)

		cpu_count = mp.cpu_count()

		num_workers = min(cpu_count, len(pairs))

		data = list(zip(X, y))
		data_subsets = self.generate_subsets(pairs, data)

		pool = mp.Pool(num_workers)
		self.svm_unit_list = pool.map(self.create_svm_unit, data_subsets)

	def one_vs_one_predict(self, X):
		labels = np.array([unit.val_to_cls_lookup(unit.predict(X)) for unit in self.svm_unit_list])
		return np.array([max(set(labels[:, idx]), key = list(labels[:, idx]).count) \
				 for idx in range(len(X))])

	def one_vs_rest_predict(self, X):
		unit_margins = np.array([unit.predict(X) for unit in self.svm_unit_list])
		best_idx = [np.argmax(unit_margins[:, idx]) for idx in range(len(X))]

		return np.array([self.svm_unit_list[best_idx[idx]].val_to_cls_lookup(unit_margins[best_idx[idx], idx]) \
				 for idx in range(len(X))])


class PCA(object):
	def __init__(self):
		pass
	
	def calc_covariance_matrix(self, X, Y):
		N = len(X)
		cov_mat = (1 / (N - 1)) * np.dot((X - np.mean(X, axis = 0)).T,  Y - np.mean(Y, axis = 0))
		return cov_mat

	def transform(self, X, num_components):
		cov_mat = self.calc_covariance_matrix(X, X)
		eig_val, eig_vec = np.linalg.eig(cov_mat)

		select_idx = eig_val.argsort()[::-1]
		eig_val = eig_val[select_idx][:num_components]
		eig_vec = np.atleast_1d(eig_vec[:, select_idx])[:, :num_components]

		X_transformed = np.dot(X, eig_vec)
		return X_transformed
