import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from scipy import signal
from sklearn import svm

def naive_detect(a, fs=200.0, win_t=0.3, axes=[0,1,2], win_bias='m', thres=None):
    if thres is None: thres = [0.5] * len(axes)
    idle_span = 0.5 * fs
    n = a.shape[1]
    b = []
    mStart = mEnd = -1
    idle = True
    for i in range(n):
        mu = False
        for j in range(len(axes)):
            if abs(a[axes[j],i]) > thres[j]:
                mu = True
                break
        if ((idle and mu) or i == n-1) and mStart > -1:
            b.append((mStart, mEnd + 1))
        if mu:
            mEnd = i
            if idle:
                mStart = i
                idle = False
        else:
            if i - mEnd > idle_span:
                idle = True
    if win_t is not None:
        win = int(win_t * fs)
        bb = []
        for i in range(len(b)):
            l, r = b[i]
            k = r - l
            if k < win:
                if win_bias == 'l':
                    r += win - k
                elif win_bias == 'm':
                    l -= (win - k) >> 1
                    r += win - (r - l)
                elif win_bias == 'r':
                    l -= win - k
            if k > win:
                r -= k - win
            if l >= 0 and r < n:
                bb.append((l, r))
    return bb

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
        acc_mean += get_acc(label[l:r], result[l:r])
        #print('fold %d: %lf' % (i, acc))
    print('cross-validation acc:', acc_mean / fold)
    return label, result

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


def plot_time(a, idx=None):
    if idx is None:
        idx = np.array(range(a.shape[0]-1)) + 1
    n = idx.shape[0]
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.plot(a[0], a[idx[i]])
    plt.show()

def plot_time_freq(a, fs, idx=None):
    if idx is None:
        idx = np.array(range(3)) + 1
    af = time2freq(a, fs)
    n = np.array(idx).shape[0]
    for i in range(n):
        plt.subplot(n*2, 1, i+1)
        plt.plot(a[0], a[idx[i]])
    for i in range(n):
        plt.subplot(n*2, 1, i+n+1)
        plt.plot(af[0], np.abs(af[idx[i]]))
    plt.show()

def plot_detail(a):
    row = 9
    pl = min(row, len(a))
    for i in range(pl):
        for j in range(6):
            plt.subplot(row, 6, i*6+j+1)
            plt.plot(a[i][j+1])
    plt.show()

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

def plot_segmentation(a, b):
    c = np.zeros((a.shape[1]))
    i = 0
    for (l,r) in b:
        c[l:r+1] = np.ones((r-l+1))*(i//10+10)
        i += 1
    plt.subplot(411)
    plt.plot(a[0], a[1])
    plt.subplot(412)
    plt.plot(a[0], a[2])
    plt.subplot(413)
    plt.plot(a[0], a[3])
    plt.subplot(414)
    plt.plot(a[0], c)
    plt.show()
