import csv
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'bold','size'   : 22}
matplotlib.rc('font', **font)

f = plt.figure(1, figsize=(10,15))
ax1 = f.add_axes([0.1, 0.66, 0.8, 0.3],
                   xticklabels=[], xlim=(0, 8))
ax2 = f.add_axes([0.1, 0.33, 0.8, 0.3],
                   xticklabels=[], xlim=(0, 8), ylim=(0,1))
ax3 = f.add_axes([0.1, 0, 0.8, 0.3],
                   xlim=(0, 8))

ax1.text(6, 0.8, 'a) Steel Indirect', fontsize=18, ha='center')
ax2.text(6, 0.8, 'b) Polyimide Indirect', fontsize=18, ha='center')
ax3.text(6, 0.8, 'c) Steel Direct', fontsize=18, ha='center')

axes = [ax1, ax2, ax3]

fs = [100000,500000,500000] #sample rate (Hz)
#steel, polyimide, abs

files = ['TEK00003.csv','TEK_KAPTON_dry.csv','TEK_abs_dry1.csv']

for k,q in enumerate(files):
    with open(q, newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)), dtype = float)
        csvfile.close
        
    spectrum = np.fft.fft(data[:,1])
    freq = (fs[k]/1000)*np.array(np.fft.fftfreq(data[:,1].shape[-1]), dtype = float)
    mag = np.sqrt(spectrum.real**2 + spectrum.imag**2)
    magmax = max(mag[int(len(mag)/2)+2:])
    mag = mag/magmax
    
    axes[k].plot(freq, mag)
    
plt.xlabel('Frequency (kHz)')
ax2.set_ylabel('Normalized Amplitude')


"""
P,_ = find_peaks(y, height=0.001)
Qx = np.zeros(len(P))
Qy = np.zeros(len(P))

for j,peak in enumerate(P):
    Qx[j] = data[:,0][peak]
    Qy[j] = y[peak]

plt.scatter(Qx,Qy, color='r')
"""
plt.show()