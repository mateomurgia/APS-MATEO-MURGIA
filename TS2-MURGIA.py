#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mateo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lfilter


def senoidal(amplitud, frec, fase, N, fs, offset=0):
    Ts = 1/fs
    tiempo = np.arange(N) * Ts
    datos = amplitud * np.sin(2*np.pi*frec*tiempo + fase) + offset
    return tiempo, datos

def ecuacion_diferencias(x):
    """Sistema: y[n] = 0.03 x[n] + 0.05 x[n-1] + 0.03 x[n-2] + 1.5 y[n-1] - 0.5 y[n-2]"""
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        x0 = x[n]
        x1 = x[n-1] if n-1 >= 0 else 0
        x2 = x[n-2] if n-2 >= 0 else 0
        y1 = y[n-1] if n-1 >= 0 else 0
        y2 = y[n-2] if n-2 >= 0 else 0
        y[n] = 0.03*x0 + 0.05*x1 + 0.03*x2 + 1.5*y1 - 0.5*y2
    return y

def energia(y):
    return np.sum(np.abs(y)**2)

def potencia(y):
    return np.mean(np.abs(y)**2)


# Señal 1: seno 2 kHz
fs1, N1 = 30000, 100
t1, sig1 = senoidal(1, 2000, 0, N1, fs1)

# Señal 2: seno 2 kHz desfasado pi/2
t2, sig2 = senoidal(np.pi/2, 2000, np.pi/2, 100, 30000)

# Señal 3: modulación (1 kHz * 2 kHz)
t3, port = senoidal(1, 1000, 0, 100, 30000)
_, mod = senoidal(1, 2000, 0, 100, 30000)
sig3 = port * mod

# Señal 4: seno 2 kHz clipeado
t4, sig4 = senoidal(1, 2000, 0, 100, 30000)
sig4_clip = np.clip(sig4, -0.75, 0.75)

# Señal 5: cuadrada 4 kHz
fs5, N5 = 100000, 200
t5 = np.arange(N5) / fs5
sig5 = signal.square(2*np.pi*4000*t5)

# Señal 6: pulso rectangular
fs6, T6 = 10000, 0.01
t6 = np.arange(0, T6, 1/fs6)
sig6 = np.zeros(len(t6))
sig6[(t6 >= 0.004) & (t6 <= 0.006)] = 1.0


# Ejercicio 1: salida del sistema LTI (sin for, señal por señal)

print("=== EJERCICIO 1 ===")

# Señal 1
y1 = ecuacion_diferencias(sig1)
tiempo_sim1 = len(sig1) / fs1
P1 = potencia(y1)
print(f"Señal 1: Fs={fs1} Hz | Tiempo={tiempo_sim1:.4f} s | Potencia={P1:.3f}")

# Señal 2
y2 = ecuacion_diferencias(sig2)
tiempo_sim2 = len(sig2) / 30000
P2 = potencia(y2)
print(f"Señal 2: Fs=30000 Hz | Tiempo={tiempo_sim2:.4f} s | Potencia={P2:.3f}")

# Señal 3
y3 = ecuacion_diferencias(sig3)
tiempo_sim3 = len(sig3) / 30000
P3 = potencia(y3)
print(f"Señal 3: Fs=30000 Hz | Tiempo={tiempo_sim3:.4f} s | Potencia={P3:.3f}")

# Señal 4
y4 = ecuacion_diferencias(sig4_clip)
tiempo_sim4 = len(sig4_clip) / 30000
P4 = potencia(y4)
print(f"Señal 4: Fs=30000 Hz | Tiempo={tiempo_sim4:.4f} s | Potencia={P4:.3f}")

# Señal 5
y5 = ecuacion_diferencias(sig5)
tiempo_sim5 = len(sig5) / fs5
P5 = potencia(y5)
print(f"Señal 5: Fs={fs5} Hz | Tiempo={tiempo_sim5:.4f} s | Potencia={P5:.3f}")

# Señal 6
y6 = ecuacion_diferencias(sig6)
tiempo_sim6 = len(sig6) / fs6
E6 = energia(y6)
print(f"Señal 6: Fs={fs6} Hz | Tiempo={tiempo_sim6:.4f} s | Energía={E6:.3f}")


# Graficar cada señal en su subplot sin usar axs.ravel()
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(t1, y1, label="Salida")
plt.plot(t1, sig1, '--', label="Entrada")
plt.title("Señal 1")
plt.legend(fontsize=8); plt.grid(alpha=0.3)

plt.subplot(3, 2, 2)
plt.plot(t2, y2, label="Salida")
plt.plot(t2, sig2, '--', label="Entrada")
plt.title("Señal 2")
plt.legend(fontsize=8); plt.grid(alpha=0.3)

plt.subplot(3, 2, 3)
plt.plot(t3, y3, label="Salida")
plt.plot(t3, sig3, '--', label="Entrada")
plt.title("Señal 3")
plt.legend(fontsize=8); plt.grid(alpha=0.3)

plt.subplot(3, 2, 4)
plt.plot(t4, y4, label="Salida")
plt.plot(t4, sig4_clip, '--', label="Entrada")
plt.title("Señal 4")
plt.legend(fontsize=8); plt.grid(alpha=0.3)

plt.subplot(3, 2, 5)
plt.plot(t5, y5, label="Salida")
plt.plot(t5, sig5, '--', label="Entrada")
plt.title("Señal 5")
plt.legend(fontsize=8); plt.grid(alpha=0.3)

plt.subplot(3, 2, 6)
plt.plot(t6, y6, label="Salida")
plt.plot(t6, sig6, '--', label="Entrada")
plt.title("Señal 6")
plt.legend(fontsize=8); plt.grid(alpha=0.3)

plt.suptitle("Salidas del sistema para las señales 1 a 6")
plt.tight_layout()
plt.show()

# Ejercicio 2


print("\n=== EJERCICIO 2 ===")

# Entrada senoidal de prueba
frecuencia_muestreo = 30000
cantidad_muestras = 300
tiempo_ej2, senoidal_entrada = senoidal(1, 1000, 0, cantidad_muestras, frecuencia_muestreo)

# Sistema A: y[n] = x[n] + 3 x[n-10]  (FIR)

coef_entrada_A = np.zeros(11)
coef_entrada_A[0] = 1.0     # coeficiente para x[n]
coef_entrada_A[10] = 3.0    # coeficiente para x[n-10]
coef_salida_A = np.array([1.0])  # no hay realimentación

salida_A = lfilter(coef_entrada_A, coef_salida_A, senoidal_entrada)

# Respuesta al impulso del sistema A
delta_A = np.zeros(len(senoidal_entrada))
delta_A[0] = 1
impulso_A = lfilter(coef_entrada_A, coef_salida_A, delta_A)

# Convolución
salida_A_conv = np.convolve(senoidal_entrada, impulso_A)[:len(senoidal_entrada)]

print(f"Sistema A (FIR): Energía={energia(salida_A):.3f}, Potencia={potencia(salida_A):.3f}")

plt.figure()
plt.plot(tiempo_ej2, salida_A, label="Salida A (lfilter)")
plt.plot(tiempo_ej2, salida_A_conv, 'o', ms=2, label="Salida A (convolución)")
plt.title("Sistema A: FIR (y[n] = x[n] + 3·x[n−10])")
plt.legend(); plt.grid(alpha=0.3); plt.show()


# Sistema B: y[n] = x[n] + 3 y[n-10]  (IIR, inestable)

coef_entrada_B = np.array([1.0])  # entrada directa
coef_salida_B = np.zeros(11)
coef_salida_B[0] = 1.0      # y[n]
coef_salida_B[10] = -3.0    # -3·y[n-10]  

salida_B = lfilter(coef_entrada_B, coef_salida_B, senoidal_entrada)

# Respuesta al impulso del sistema B
impulso_B = lfilter(coef_entrada_B, coef_salida_B, delta_A)

# Convolución con respuesta truncada
salida_B_conv = np.convolve(senoidal_entrada, impulso_B)[:len(senoidal_entrada)]

print(f"Sistema B (IIR inestable): Energía≈{energia(salida_B):.3e}, Potencia≈{potencia(salida_B):.3e}")

plt.figure()
plt.plot(tiempo_ej2, salida_B, label="Salida B (IIR lfilter)")
plt.plot(tiempo_ej2, salida_B_conv, 'o', ms=2, label="Salida B (convolución trunc.)")
plt.title("Sistema B: IIR (y[n] = x[n] + 3·y[n−10])")
plt.legend(); plt.grid(alpha=0.3); plt.show()
