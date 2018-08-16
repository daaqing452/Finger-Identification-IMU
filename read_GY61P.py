# -*- coding: utf-8 -*-
import binascii
import datetime
import serial
import serial.tools.list_ports
import time

LOG = True
STRIDE = 3

def int8(x):
	if x < 128:
		return x
	else:
		return x - 256

# acc_port = 'COM3'
# s = serial.Serial(port=acc_port, baudrate=115200)

port_list = list(serial.tools.list_ports.comports())
print(port_list[1])
print(port_list[2])
s = serial.Serial(port_list[1][0], baudrate=115200)
if LOG: datafile = open(datetime.datetime.now().strftime('data/acc61P-%Y%m%d-%H%M%S')+'.txt','w')

cnt = 0
while True:
	cnt += 1
	raw = s.read(1)
	if raw[0] != 0x55:
		print(raw[0])
		continue
	raw = b'' + raw + s.read(10)
	
	# acceleration
	elif raw[1] == 0x51:
		ax = ((int8(raw[3]) << 8) | raw[2]) / 32768.0 * 16 * 9.8
		ay = ((int8(raw[5]) << 8) | raw[4]) / 32768.0 * 16 * 9.8
		az = ((int8(raw[7]) << 8) | raw[6]) / 32768.0 * 16 * 9.8
	# angular velocity
	elif raw[1] == 0x52:
		rx = ((int8(raw[3]) << 8) | raw[2]) / 32768.0 * 2000
		ry = ((int8(raw[5]) << 8) | raw[4]) / 32768.0 * 2000
		rz = ((int8(raw[7]) << 8) | raw[6]) / 32768.0 * 2000
	# angle
	elif raw[1] == 0x53:
		roll  = ((int8(raw[3]) << 8) | raw[2]) / 32768.0 * 180
		pitch = ((int8(raw[5]) << 8) | raw[4]) / 32768.0 * 180
		yaw   = ((int8(raw[7]) << 8) | raw[6]) / 32768.0 * 180
	# quaternion
	elif raw[1] == 0x59:
		q0 = ((int8(raw[3]) << 8) | raw[2]) / 32768.0
		q1 = ((int8(raw[5]) << 8) | raw[4]) / 32768.0
		q2 = ((int8(raw[7]) << 8) | raw[6]) / 32768.0
		q3 = ((int8(raw[9]) << 8) | raw[8]) / 32768.0

	if cnt % STRIDE == 0:
		if LOG: datafile.write(str(time.time()) + ' ' + str(ax) + ' ' + str(ay) + ' ' + str(az) + '\n')
		if cnt % 50 == 0:
			print(cnt, ax, ay, az)