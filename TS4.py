import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp

def eje_temporal (N, fs):
    
    Ts = 1/fs
    t_final = N * Ts
    tt = np.arange (0, t_final, Ts)
    return tt


def func_senoidal (tt, frec, amp, fase = 0, v_medio = 0):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx
R=200
frec_rand2 = np.random.uniform(-2,2)
SNR = 10 # SNR en dB
amp_0 = np.sqrt(2) # amplitud en V
N = 1000
fs = 1000
df = fs / N # Hz, resolución espectral

nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral
tt = eje_temporal (N = N, fs = fs)

s_1 = func_senoidal (tt = tt, amp = amp_0, frec = N/4 + frec_rand2*df)
tt_re = tt.reshape(N,1)
frec_rand_re = frec_rand2.reshape(1,R)

X=np.tile(tt_re,)
pot_ruido = amp_0**2 / (2*10**(SNR/10))        
print (f"Potencia de SNR {pot_ruido:3.1f}")   
                      
ruido = np.random.normal (0, np.sqrt(pot_ruido), N)
var_ruido = np.var (ruido)
print (f"Potencia de ruido -> {var_ruido:3.3f}")

x_1 = s_1 + ruido  # modelo de señal

X_1ruido = (1/N)*fft (x_1)
# print (np.var(x_1))


plt.plot (ff, 10*np.log10(2*np.abs(X_1ruido)**2), color='orange', label='X_1')
plt.xlim((0,fs/2))
plt.grid (True)
plt.legend ()
plt.show ()