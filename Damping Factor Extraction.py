import csv
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

#############################
# FFT
f = plt.figure(1, figsize=(10,7))
#############################

fs = 100000 #sample rate (Hz)
#steel: 100000
#polyimide: 500000
#abs:500000

with open('TEK00003.csv', newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)), dtype = float)
    csvfile.close
    
spectrum = np.fft.fft(data[:,1])
freq = (fs/1000)*np.array(np.fft.fftfreq(data[:,1].shape[-1]), dtype = float)
mag = np.sqrt(spectrum.real**2 + spectrum.imag**2)

plt.plot(freq, mag)
plt.xlim((1,8))
plt.ylim((0, 30))
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude')

############################
# FILTERING
f1 = plt.figure(2, figsize=(10,7))
############################

i = 0
bands = [[350,700],[1300,1800],[3600,4200]]
natural = [718, 2830, 6463]

nyq = 0.5*fs
low = bands[i][0]/nyq
high = bands[i][1]/nyq
b, a = butter(3, [low, high], btype='band')
y = lfilter(b,a,data[:,1])

##############################
# MODAL DAMPENING FACTOR
##############################
P,_ = find_peaks(y, height=0.001)
Qx = np.zeros(len(P))
Qy = np.zeros(len(P))

for j,peak in enumerate(P):
    Qx[j] = data[:,0][peak]
    Qy[j] = y[peak]

plt.plot(data[:,0], y)
plt.scatter(Qx,Qy, color='r')
plt.xlabel('Time (s)')
plt.ylabel('Bandpass Signal (arb.)')

slope,intercept = np.polyfit(Qx[2:],np.log(Qy[2:]),1) #extract damping factor
#print(-slope/natural[i])

linear = np.zeros(len(data[:,0]), dtype = float)
exponential = np.zeros(len(data[:,0]), dtype = float)
print(slope)
print(intercept)

for i in range(len(data[:,0])):
    linear[i] = slope*data[:,0][i] + intercept
    exponential[i] = np.exp(intercept)*np.exp(slope*data[:,0][i])

plt.plot(data[:,0],exponential)

f2 = plt.figure(3, figsize=(10,7))
plt.scatter(Qx[2:],np.log(Qy[2:]), color='r')
plt.plot(data[:,0], linear)
plt.xlabel('Time (s)')
plt.ylabel('linearized data (arb.)')

plt.show()
