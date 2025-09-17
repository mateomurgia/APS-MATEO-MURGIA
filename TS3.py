# -*- coding: utf-8 -*-
"""
TP3 - Incisos a, b y c
Autor: Mateo
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# Parámetros
fs = 1000      # Frecuencia de muestreo
N = 1000       # Número de muestras
Ts = 1/fs
deltaF = fs/N
tt = np.arange(N)*Ts
freqs = np.fft.fftfreq(N, Ts)  # Eje de frecuencias

# --------------------
# Función senoidal normalizada (varianza unitaria)
# --------------------
def mi_funcion_sen(f, N, fs, a0=1, fase=0):
    Ts = 1/fs
    tt = np.arange(N) * Ts
    x = a0 * np.sin(2*np.pi*f*tt + fase)
    x = x - x.mean()                     # quitar media
    x = x / np.std(x)                    # normalizar varianza = 1
    return tt, x

# Señales: fs/4, fs/4+0.25 y fs/4+0.5 (en Δf)
tt, x1 = mi_funcion_sen(f = (N/4) * deltaF, N=N, fs=fs)
tt, x2 = mi_funcion_sen(f = ((N/4) + 0.25) * deltaF, N=N, fs=fs)
tt, x3 = mi_funcion_sen(f = ((N/4) + 0.5) * deltaF, N=N, fs=fs)

# FFTs
X1, X2, X3 = fft(x1), fft(x2), fft(x3)
X1abs, X2abs, X3abs = 1/N * np.abs(X1), 1/N * np.abs(X2), 1/N * np.abs(X3)

# --------------------
# Inciso a) Graficar PSDs
# --------------------
plt.figure()
plt.title("Inciso a) FFT y densidad espectral de potencia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.grid(True)
plt.plot(freqs, 10*np.log10(2*X1abs**2), label="fs/4")
plt.plot(freqs, 10*np.log10(2*X2abs**2), label="fs/4 + 0.25")
plt.plot(freqs, 10*np.log10(2*X3abs**2), label="fs/4 + 0.5")
plt.xlim((0, fs/2))
plt.legend()
plt.show()

# --------------------
# Inciso b) Verificación de Parseval
# --------------------
def verificar_parseval(x, X, N, nombre=""):
    A = np.sum(np.abs(x)**2) / N        # potencia en el tiempo
    B = np.sum(np.abs(X)**2) / N**2     # potencia en frecuencia
    dif = A - B
    print(f"\nSeñal {nombre}:")
    print(f" Potencia tiempo = {A:.5f}")
    print(f" Potencia frec   = {B:.5f}")
    print(f" Diferencia (A-B) = {dif:.2e}")
    
    if np.isclose(A, B, rtol=1e-10, atol=1e-12):
        print(" ✅ Se cumple Parseval")
    else:
        print(" ❌ No se cumple Parseval")

verificar_parseval(x1, X1, N, "fs/4")
verificar_parseval(x2, X2, N, "fs/4 + 0.25")
verificar_parseval(x3, X3, N, "fs/4 + 0.5")

# --------------------
# Inciso c) Zero Padding a las tres señales
# --------------------
def aplicar_zero_padding(x, N, factor=10):
    zeros = np.zeros(len(x)*(factor-1))       # agrego ceros
    xPadding = np.concatenate((x, zeros))     # señal extendida
    XPadding = fft(xPadding)                  # FFT
    Npad = len(xPadding)
    freqs_pad = np.fft.fftfreq(Npad, 1/fs)    # eje de frecuencias
    return xPadding, XPadding, freqs_pad, Npad

# Aplicar zero padding a las tres señales
x1_pad, X1_pad, freqs_pad, Npad = aplicar_zero_padding(x1, N, factor=10)
x2_pad, X2_pad, freqs_pad, Npad = aplicar_zero_padding(x2, N, factor=10)
x3_pad, X3_pad, freqs_pad, Npad = aplicar_zero_padding(x3, N, factor=10)

# Gráficos comparativos
plt.figure()
plt.title("Zero Padding - Señal fs/4")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.grid(True)
plt.plot(freqs_pad, 20*np.log10(np.abs(X1_pad)/Npad), label="Con Zero Padding")
plt.plot(freqs, 20*np.log10(np.abs(X1)/N), "--", label="Original")
plt.xlim((0, fs/2))
plt.legend()

plt.figure()
plt.title("Zero Padding - Señal fs/4 + 0.25")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.grid(True)
plt.plot(freqs_pad, 20*np.log10(np.abs(X2_pad)/Npad), label="Con Zero Padding")
plt.plot(freqs, 20*np.log10(np.abs(X2)/N), "--", label="Original")
plt.xlim((0, fs/2))
plt.legend()

plt.figure()
plt.title("Zero Padding - Señal fs/4 + 0.5")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.grid(True)
plt.plot(freqs_pad, 20*np.log10(np.abs(X3_pad)/Npad), label="Con Zero Padding")
plt.plot(freqs, 20*np.log10(np.abs(X3)/N), "--", label="Original")
plt.xlim((0, fs/2))
plt.legend()

plt.show()
