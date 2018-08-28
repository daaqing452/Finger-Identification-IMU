import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from scipy import signal
from sklearn import svm

def get_accuracy(label, result):
    n = label.shape[0]
    wa_index = np.nonzero(label != result)[0]
    wa = len(wa_index)
    acc = 100 * (1 - wa/(1.0*n))
    return acc

def cross_validation(data, label, clf=svm.SVC(), fold=10):
    n = data.shape[0]
    arr = np.arange(n)
    np.random.shuffle(arr)
    data = np.array([data[i] for i in arr])
    label = np.array([label[i] for i in arr])
    acc_mean = 0
    result = np.zeros((n))
    for i in range(fold):
        l = int(n * i / fold)
        r = int(n * (i+1) / fold)
        data2 = np.concatenate((data[:l], data[r:]))
        label2 = np.concatenate((label[:l], label[r:]))
        clf.fit(data2, label2)
        result[l:r] = clf.predict(data[l:r])
        acc = get_accuracy(label[l:r], result[l:r])
        acc_mean += acc
        print('\tfold %d: %lf' % (i, acc))
    print('cross-validation acc:', acc_mean / fold)
    return result

def time2freq(a, fs, segment=None):
    n = a.shape[1]
    af = np.fft.fft(a, axis=1)
    af[0] = np.arange(n) * fs / n
    if segment is not None:
        af = af[:, int(segment[0] / fs * n):int(segment[1] / fs * n)]
    return af

def highpass_filter(a, btc=0.4):
    coeff_b, coeff_a = signal.butter(3, btc, 'highpass')
    return signal.filtfilt(coeff_b, coeff_a, a, axis=0)

def kalman_filter(obs, q=0.01):
    n = obs.shape[0]
    x = np.zeros((n))
    x[0] = obs[0]
    p = q
    for i in range(1, n):
        k = math.sqrt(p * p + q * q)
        h = math.sqrt(k * k / (k * k + q * q))
        x[i] = obs[i] * h + x[i-1] * (1 - h)
        p = math.sqrt((1 - h) * k * k)
    return x


def plot_confusion(a, b, confusion_show=True, plot_show=True):
    a = np.array(a, dtype=int)
    b = np.array(b, dtype=int)
    n = 0
    for i in range(a.shape[0]):
        n = max(n, a[i]+1)
        n = max(n, b[i]+1)
    c = np.zeros((n,n))
    for i in range(a.shape[0]):
    	c[a[i], b[i]] += 1
    if confusion_show:
        print(c)
    if plot_show:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = ax.imshow(c, interpolation='nearest')
        cb = fig.colorbar(res)
        plt.show()
    return c

def plot_label_acc(label, result, show=True):
    c = plot_confusion(label, result, confusion_show=False, plot_show=False)
    if show:
        print('label\tprecise\trecall')
    ps = []
    rs = []
    for i in range(c.shape[0]):
        precision = c[i, i] / c[:,i].sum()
        recall = c[i, i] / c[i].sum()
        ps.append(precision)
        rs.append(recall)
        if show:
            print('%d\t%.3lf\t%.3lf' % (i, precision*100, recall*100))
    return ps, rs


def int8(x):
    if x < 128:
        return x
    else:
        return x - 256

def read_JY901(s, acc, gyro):
    print('enter0')
    cnt = 0
    while True:
        cnt += 1
        raw = s.read(1)
        if raw[0] != 0x55: continue
        raw = b'' + raw + s.read(10)
        # acceleration
        if raw[1] == 0x51:
            acc[0] = [(int8(raw[3]) << 8) | raw[2], (int8(raw[5]) << 8) | raw[4], (int8(raw[7]) << 8) | raw[6]]
            acc[0] = np.array(acc[0], dtype=float) / 32768.0 * 16
        # angular velocity
        elif raw[1] == 0x52:
            gyro[0] = [(int8(raw[3]) << 8) | raw[2], (int8(raw[5]) << 8) | raw[4], (int8(raw[7]) << 8) | raw[6]]
            gyro[0] = np.array(gyro[0], dtype=float) / 32768.0 * 2000
        # angle
        elif raw[1] == 0x53:
            rotation = [(int8(raw[3]) << 8) | raw[2], (int8(raw[5]) << 8) | raw[4], (int8(raw[7]) << 8) | raw[6]]
            rotation = np.array(rotation, dtype=float) / 32768.0 * 180
        # quaternion
        elif raw[1] == 0x59:
            quaternion = [(int8(raw[3]) << 8) | raw[2], (int8(raw[5]) << 8) | raw[4], (int8(raw[7]) << 8) | raw[6], (int8(raw[9]) << 8) | raw[8]]
            quaternion = np.array(quaternion, dtype=float) / 32768.0

def read_GY9250(s, acc, gyro):
    print('enter1')
    s.write(b'\x01')
    cnt = 0
    while True:
        data = s.read(14)
        # print(type(data))
        num = []
        for i in range(7):
            num.append(int((data[i*2]<<8) | data[i*2+1]))
            if num[i] > 2**15: num[i] -= 2**16
        acc[1] = np.array(num[0:3]) / 32768.0 * 16
        ther = num[3:4]
        gyro[1] = np.array(num[4:7]) / 32768.0 * 2000
        cnt += 1
