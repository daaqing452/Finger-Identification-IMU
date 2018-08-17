import numpy as np
from sklearn import svm
from utils import *

def read(f):
	a = []
	b = []
	c = []
	idx = 0
	while True:
		line = f.readline()
		if len(line) == 0: break
		arr = line[:-1].split(' ')
		if int(arr[1]) == 1: b.append(idx)
		c.append(int(arr[1]))
		a.append([float(arr[i]) for i in range(2, 14)])
		idx += 1
	return np.array(a).T, b, np.array(c)

def feature(x):
	def feature_accelerometer(f, x, norm=False):
		if norm: x = (x - x.mean()) / x.std()
		f.append(x.max())
		f.append(x.min())
		f.append(x.max() / x.min())
		i = x.argmax()
		j = x.argmin()
		f.append(i < j)

	def feature_gyrometer(f, x):
		return feature_accelerometer(f, x)

	f = []
	for i in range(0, 6): feature_accelerometer(f, a[i])
	for i in range(6, 12): feature_accelerometer(f, a[i])
	return f

filename = sys.argv[1]
f = open(filename, 'r')
a, b, c = read(f)
print('#' * 3)

for i in range(a.shape[0]):
	a[i] = highpass_filter(a[i])
	a[i] = kalman_filter(a[i])

datas = []
labels = []
for i in range(len(b)):
	raw = a[b[i]-30 : b[i]+30]
	label = i % 5
	data = feature(raw)
	datas.append(data)
	labels.append(labels)
datas = np.array(datas)
labels = np.array(labels)

cross_validation(datas, labels)