import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from utils import *

op = []

def read(f):
	a = []
	while True:
		line = f.readline()
		if len(line) == 0: break
		arr = line[:-1].split(' ')
		op.append(int(arr[1]))
		a.append([float(arr[i]) for i in range(2, 14)])
	return np.array(a)

filename = sys.argv[1]
f = open(filename, 'r')
a = read(f).T
f.close()

for i in range(a.shape[0]):
	a[i] = highpass_filter(a[i])

b = naive_detect(a, axes=[0,1,2,6,7,8], thres=[0.05, 0.05, 0.05, 5, 5, 5])
print(len(b))

c = np.zeros((a.shape[1]))
for (l,r) in b:
	for j in range(l, r):
		c[j] = 1

plt.figure(1)
plt.subplot(611)
plt.plot(a[0])
plt.plot(op)
plt.subplot(612)
plt.plot(a[1])
plt.plot(c)
plt.subplot(613)
plt.plot(a[2])
plt.subplot(614)
plt.plot(a[6])
plt.subplot(615)
plt.plot(a[7])
plt.subplot(616)
plt.plot(a[8])

plt.figure(2)
plt.subplot(611)
plt.plot(a[3])
plt.subplot(612)
plt.plot(a[4])
plt.subplot(613)
plt.plot(a[5])
plt.subplot(614)
plt.plot(a[9])
plt.subplot(615)
plt.plot(a[10])
plt.subplot(616)
plt.plot(a[11])

plt.show()