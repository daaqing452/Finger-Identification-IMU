# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pyHook
import pythoncom
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
import threading
import time
from sklearn import svm
from sklearn.externals import joblib
from utils import *

# configuration
FPS = 200
SPAN = 60

# global acceleration & gyroscope
acc = [[0, 0, 0], [0, 0, 0]]
gyro = [[0, 0, 0], [0, 0, 0]]

# get serial
port_list = list(serial.tools.list_ports.comports())
print(port_list[1])
print(port_list[2])
FIRST = 2
serials = [ [port_list[FIRST][0], 115200, read_JY901], [port_list[3-FIRST][0], 115200, read_GY9250] ]

# set up thread to read serial
threads = []
for i in range(len(serials)):
	s = serial.Serial(serials[i][0], serials[i][1])
	t = threading.Thread(target=serials[i][2], args=(s, acc, gyro))
	t.setDaemon(True)
	t.start()
	threads.append(t)

clf = joblib.load(sys.argv[1])

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

# update per frame
def updatePerFrame():
	global acc, gyro, f, key_press

	history = []
	cnt = 0
	counter = 0
	
	while True:
		cnt += 1

		# get sensing information
		now = []
		for i in range(6):
			now.append(acc[i // 3][i % 3])
		for i in range(6):
			now.append(gyro[i // 3][i % 3])
		history.append(now)
		if cnt % FPS == 0: print(now)

		if counter == 1:
			raw = np.array(history[tapEventIndex-SPAN//2 : tapEventIndex+SPAN//2]).T
			data = [feature(raw)]
			result = clf.predict(data)
			print()
			print(result)
			print()
			history = history[tapEventIndex:]

		counter = max(counter - 1, 0)
		if int(key_press) == 1:
			print('tap')
			key_press = 0
			counter = 40
			tapEventIndex = len(history)

		# control frequency
		time.sleep(1.0 / FPS)


# keyboard event
def onKeyboardEvent(event):
	global key_press
	if event.Ascii == 32:
		key_press = True
	return 0

def listenKeyboard():
	hm = pyHook.HookManager()
	hm.KeyDown = onKeyboardEvent
	hm.HookKeyboard()
	pythoncom.PumpMessages()

key_press = False


# t = threading.Thread(target=updatePerFrame, args=())
t = threading.Thread(target=listenKeyboard, args=())
t.setDaemon(True)
t.start()

# listenKeyboard()
updatePerFrame()