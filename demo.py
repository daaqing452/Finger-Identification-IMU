# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pyHook
import pythoncom
import serial
import serial.tools.list_ports
import threading
import time
from utils import *

# configuration
FRAME = 200
FPS = 200

# global acceleration & gyroscope
acc = [[0, 0, 0], [0, 0, 0]]
gyro = [[0, 0, 0], [0, 0, 0]]

# get serial
port_list = list(serial.tools.list_ports.comports())
print(port_list[1])
print(port_list[2])
serials = [ [port_list[2][0], 115200, read_JY901], [port_list[1][0], 115200, read_GY9250] ]

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
	cnt = 0
	tapCnt = 0
	while True:
		cnt += 1

		# get sensing information
		for i in range(6):
			a[i].append(acc[i // 3][i % 3])
			g[i].append(gyro[i // 3][i % 3])

		if cnt % FRAME == 0:
			print(acc[0], acc[1], gyro[0], gyro[1])

		# control frequency
		time.sleep(1.0 / FPS)


# keyboard event
def onKeyboardEvent(event):
	global key_press
	if event.Ascii == 32:
		key_press = True
	return 0

hm = pyHook.HookManager()
hm.KeyDown = onKeyboardEvent
hm.HookKeyboard()
key_press = False

# start update per frame
t = threading.Thread(target=updatePerFrame, args=())
t.setDaemon(True)
t.start()

# blocked listen
pythoncom.PumpMessages()