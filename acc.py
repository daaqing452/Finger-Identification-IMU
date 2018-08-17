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

# configuration
LOG = True
PLOT = False
FRAME = 200
FPS = 200

# global acceleration & gyroscope
acc = [[0, 0, 0], [0, 0, 0]]
gyro = [[0, 0, 0], [0, 0, 0]]

# log file
if LOG:
	f = open(datetime.datetime.now().strftime('data/acc-%Y%m%d-%H%M%S')+'.txt','w')
else:
	f = None

# convert byte to uint8 when reading serial
def int8(x):
	if x < 128:
		return x
	else:
		return x - 256

def read_JY901(s):
	print('enter0')
	global acc0
	global gyro0
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

def read_GY9250(s):
	print('enter1')
	global acc1
	global gyro1
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

# get serial
port_list = list(serial.tools.list_ports.comports())
print(port_list[1])
print(port_list[2])
serials = [ [port_list[1][0], 115200, read_JY901], [port_list[2][0], 115200, read_GY9250] ]

# set up thread to read serial
threads = []
for i in range(len(serials)):
	s = serial.Serial(serials[i][0], serials[i][1])
	t = threading.Thread(target=serials[i][2], args=(s,))
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

		# plot
		if PLOT and len(a[0]) > FRAME and cnt % FRAME == 0:
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