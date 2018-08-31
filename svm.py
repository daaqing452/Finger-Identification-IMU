import numpy as np
import pickle
from sklearn import svm, neighbors
from sklearn.externals import joblib
from utils import *

SPAN = 60

def feature(x):
	def feature_accelerometer(f, x, norm=True):
		if norm: x = (x - x.mean()) / x.std()
		f.append(x.max())
		f.append(x.min())
		# f.append(x.max() / x.min())
		i = x.argmax()
		j = x.argmin()
		# f.append(i < j)

	def feature_gyrometer(f, x):
		return feature_accelerometer(f, x)

	f = []
	for i in range(0, 6): feature_accelerometer(f, x[i])
	for i in range(6, 12): feature_accelerometer(f, x[i])
	return f

def make_data_label(a, b):
	datas = []
	labels = []
	for i in range(len(b)):
		if i % 100 == 0: print('\t', i)
		raw = a[:, b[i]-SPAN//2 : b[i]+SPAN//2]
		label = i % 5
		data = feature(raw)
		datas.append(data)
		labels.append(label)
	datas = np.array(datas)
	labels = np.array(labels)
	return datas, labels

optype = sys.argv[1]

if optype == 'train':
	filename = sys.argv[2]
	f = open(filename, 'r')
	a, b = pickle.load(open(filename, 'rb'))
	print('######### finish loading #########')

	for i in range(a.shape[0]):
		# a[i] = highpass_filter(a[i])
		# a[i] = kalman_filter(a[i])
		pass
	print('######## finish filtering ########')

	datas, labels = make_data_label(a, b)
	print('###### finish featurization ######')

	clf = neighbors.KNeighborsClassifier(3)
	cross_validation(datas, labels, clf)

	um = np.zeros((5, 24, 200))
	for i in range(datas.shape[0]):
		data = datas[i]
		label = labels[i]
		um[label, :, i:i+1] = data.reshape((24, 1))
	for j in range(24):
		s = '['
		for i in range(5):
			l = um[i,j].mean() - um[i,j].std()
			r = um[i,j].mean() + um[i,j].std()
			s += '('
			s += str(int(um[i,j].mean() * 100) / 100.0)
			s += ','
			s += str(int(l * 100) / 100.0)
			s += ','
			s += str(int(r * 100) / 100.0)
			s += ')\t'
		s += ']'
		print(j, s)

	clf = svm.SVC()
	clf.fit(datas, labels)
	results = clf.predict(datas)
	print(get_accuracy(labels, results))
	joblib.dump(clf, 'model.m')

if optype == 'test':
	filename = sys.argv[2]
	f = open(filename, 'r')
	a, b = pickle.load(open(filename, 'rb'))

	datas, labels = make_data_label(a, b)
	labels = 4 - labels
	clf = joblib.load(sys.argv[3])
	results = clf.predict(datas)
	print(get_accuracy(labels, results))
	print(labels)
	print(results)