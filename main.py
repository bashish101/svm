import numpy as np
from svm import SVM, PCA
from sklearn import svm, datasets

def iris_classification():
	print('\nIris classification using SVM\n')
	print('Initiating Data Load...')
	iris = datasets.load_iris()
	X, y = iris.data, iris.target

	size = len(X)
	indices = list(range(size))
	np.random.shuffle(indices)
	X, y = np.array([X[idx] for idx in indices]), np.array([y[idx] for idx in indices])

	train_size = int(0.8 * len(X))
	X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

	print('Data load complete!')

	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(X_train, y_train)
	predictions = classifier.predict(X_test)
	accuracy = np.sum(predictions == y_test) / len(predictions)
	print("Accuracy = {:.2f} on sklearn classifier".format(accuracy))

	print('Constructing SVM classifier...')
	classifier = SVM(kernel = 'linear', C = 0.1, strategy = 'one_vs_rest')
	classifier.fit(X_train, y_train)

	print('Generating test predictions...')
	predictions = classifier.predict(X_test)
	accuracy = np.sum(predictions == y_test) / len(predictions)
	print("Accuracy = {:.2f} on custom classifier".format(accuracy))

def digit_recognition():
	print('\nDigit recognition using SVM\n')
	print('Initiating Data Load...')
	digits = datasets.load_digits()
	X, y = digits.data, digits.target

	pca = PCA()
	X = pca.transform(X, num_components = 10)

	size = len(X)
	indices = list(range(size))
	np.random.shuffle(indices)
	X, y = np.array([X[idx] for idx in indices]), np.array([y[idx] for idx in indices])

	train_size = int(0.8 * len(X))
	X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(X_train, y_train)
	predictions = classifier.predict(X_test)
	accuracy = np.sum(predictions == y_test) / len(predictions)	
	print("Accuracy = {:.2f} on sklearn classifier".format(accuracy))

	print('Constructing SVM classifier...')
	classifier = SVM(kernel = 'linear')
	classifier.fit(X_train, y_train)

	print('Generating test predictions...')
	predictions = classifier.predict(X_test)
	accuracy = np.sum(predictions == y_test) / len(predictions)
	print("Accuracy = {:.2f} on custom classifier".format(accuracy))

def letter_recognition():
	print('\nLetter recognition using SVM\n')
	print('Initiating Data Load...')
	data = np.loadtxt('./data/letter-recognition.data',
			  dtype= 'float32', 
			  delimiter = ',',
			  converters= {0: lambda ch: ord(ch) - ord('A')})
	
	train_data, test_data = np.vsplit(data, 2)

	train_labels, train_features = np.hsplit(train_data, [1])
	test_labels, test_features = np.hsplit(test_data, [1])

	train_labels = train_labels.squeeze()
	test_labels = test_labels.squeeze()

	select_idx = np.isin(train_labels, list(range(5))) 
	train_labels, train_features = train_labels[select_idx], train_features[select_idx]

	select_idx = np.isin(test_labels, list(range(5))) 
	test_labels, test_features = test_labels[select_idx], test_features[select_idx]

	print('Data load complete!')

	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(train_features, train_labels)
	predictions = classifier.predict(test_features)
	accuracy = np.sum(predictions == test_labels) / len(predictions)
	print("Accuracy = {:.2f} on sklearn classifier".format(accuracy))

	print('Constructing SVM classifier...')
	classifier = SVM(kernel = 'rbf', C = 0.1)
	classifier.fit(train_features, train_labels)

	print('Generating test predictions...')
	predictions = classifier.predict(test_features)
	accuracy = np.sum(predictions == test_labels) / len(predictions)

	print("Accuracy = {:.2f} on custom classifier".format(accuracy))


if __name__ == '__main__':
	np.random.seed(3)

	iris_classification()
	digit_recognition()
	# letter_recognition()
