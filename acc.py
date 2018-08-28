# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pyHook
import pythoncom
import serial
import serial.tools.list_ports
import datetime
import time
import threading
from utils import *

# configuration
LOG = True
PLOT = False
FPS = 200

# global acceleration & gyroscope
acc = [[0, 0, 0], [0, 0, 0]]
gyro = [[0, 0, 0], [0, 0, 0]]

# log file
if LOG:
	f = open(datetime.datetime.now().strftime('data/acc-%Y%m%d-%H%M%S')+'.txt','w')
else:
	f = None

# get serial
port_list = list(serial.tools.list_ports.comports())
print(port_list[1])
print(port_list[2])
serials = [ [port_list[1][0], 115200, read_JY901], [port_list[2][0], 115200, read_GY9250] ]

# set up thread to read serial
threads = []
for i in range(len(serials)):
	s = serial.Serial(serials[i][0], serials[i][1])
	t = threading.Thread(target=serials[i][2], args=(s, acc, gyro))
	t.setDaemon(True)
	t.start()
	threads.append(t)

# update per frame
def updatePerFrame():
	global acc, gyro, f, key_press

	a = [[], [], [], [], [], []]
	g = [[], [], [], [], [], []]
	plt.ion()
	tick = 0
	cnt = 0
	tapCnt = 0
	while True:
		# control frequency
		'''tick = time.time()
		span = tick - last_tick
		if span < 1.0 / FPS: continue
		last_tick = tick'''

		cnt += 1

		# get sensing information
		for i in range(6):
			a[i].append(acc[i // 3][i % 3])
			g[i].append(gyro[i // 3][i % 3])

		if cnt % FPS == 0:
			print(acc[0], acc[1], gyro[0], gyro[1])

		# plot
		if PLOT and len(a[0]) > FPS and cnt % FPS == 0:
			for i in range(6):
				plt.subplot(6, 2, i*2+1)
				plt.cla()
				plt.subplot(6, 2, i*2+2)
				plt.cla()
			for i in range(6):
				plt.subplot(6, 2, i*2+1)
				plt.plot(a[i])
				plt.subplot(6, 2, i*2+2)
				plt.plot(g[i])
			plt.pause(0.001)

		# log file
		if LOG:
			f.write(datetime.datetime.now().strftime("%H:%M:%S.%f") + ' ')
			f.write(str(int(key_press)) + ' ')
			if key_press == True:
				key_press = False
				tapCnt += 1
				print('tap ', tapCnt)
			f.write(str(acc[0][0]) + ' ' + str(acc[0][1]) + ' ' + str(acc[0][2]) + ' ')
			f.write(str(acc[1][0]) + ' ' + str(acc[1][1]) + ' ' + str(acc[1][2]) + ' ')
			f.write(str(gyro[0][0]) + ' ' + str(gyro[0][1]) + ' ' + str(gyro[0][2]) + ' ')
			f.write(str(gyro[1][0]) + ' ' + str(gyro[1][1]) + ' ' + str(gyro[1][2]) + '\n')

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