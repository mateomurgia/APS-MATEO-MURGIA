# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 18:40:40 2025

@author: mateo
"""

import numpy as np 
import matplotlib.pyplot as plt
from numpy.fft import fft

N=1000
fs= 1000
df= fs/N
ts=1/fs

def sen(ff,nn,amp=1,dc=0,ph=0,fs=2):
    N=np.arange(nn)
    t=N/fs
    x= dc + amp*np.sin(2*np.pi*ff*t+ph)
    return t,x

t1,x1=sen(ff=(N/4)*df,nn=N, fs=fs)
t2,x2=sen(ff=((N/4)+1)*df,nn=N, fs=fs)
_,x3=sen(ff=((N/4)+0.5)*df,nn=N, fs=fs)


#calculo de modulo y fase de las ffts         
X1=fft(x1)
X1_ABS=np.abs(X1)
X1_ANGLE=np.angle(X1)

X2=fft(x2)
X2_ABS=np.abs(X2)
X2_ANGLE=np.angle(X2)

X3=fft(x3)
X3_ABS=np.abs(X3)
X3_ANGLE=np.angle(X3)




eje_freqs=np.arange(N)*df
plt.figure(1)
#plt.plot(eje_freqs,X1_ABS,'x',label='Modulo X1')
#plt.plot(eje_freqs,X2_ABS,'o',label='Modulo X2')
#plt.plot(eje_freqs,X3_ABS,'o',label='Modulo X3')

plt.plot(eje_freqs,np.log10(X1_ABS)*20,'x',label='Modulo X1')
plt.plot(eje_freqs,np.log10(X2_ABS)*20,'o',label='Modulo X2')
plt.plot(eje_freqs,np.log10(X3_ABS)*20,'o',label='Modulo X3')
plt.xlim([0,fs/2])
plt.legend()
plt.title('FFT')
plt.xlabel('Tiempo')
plt.ylabel('Frecuencia (dB)')


plt.grid()
plt.show()
# %%
#identificar var(x)=1
import numpy as np 
import matplotlib.pyplot as plt
from numpy.fft import fft

N = 1000       
fs = 1000      
df = fs/N      
ts = 1/fs

def sen(ff, nn, amp=np.sqrt(2), dc=0, ph=0, fs=2):
    n = np.arange(nn)
    t = n/fs
    x = dc + amp*np.sin(2*np.pi*ff*t + ph)
    return t, x

#
ff = (N/4)*df
amp = np.sqrt(2)  

t1, x1 = sen(ff=ff, nn=N, fs=fs, amp=amp)

varianza = np.var(x1)

print(f"Amplitud usada = {amp:.5f}")
print(f"Varianza medida = {varianza:.5f}")
eje_freqs=np.arange(N)*df
plt.figure(1)


X1=fft(x1)
X1_ABS=np.abs(X1)
X1_AL_CUADRADO=X1_ABS**2

plt.figure(2)
plt.plot(eje_freqs,np.log10(X1_AL_CUADRADO)*10,'x',label='Modulo frec. espectral')
plt.legend()


