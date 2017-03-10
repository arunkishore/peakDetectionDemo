%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 128
# sample spacing
T = 1.0 / 128
x = np.linspace(0.0, N*T, N)
y = y#X_train.iloc[7350]  
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()


# which data to use?
# An example to get started with the low pass filter
# An example for FFT analysis?
