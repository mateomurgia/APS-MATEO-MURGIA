import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

def func_senoidal (a_max, frec, fase, cant_muestras, frec_muestreo, v_medio):
    
    Ts = 1/frec_muestreo
    t_final = cant_muestras * Ts
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    xx = a_max * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    
    return tt, xx
#SEÑAL 1
a_max = 0
frec = 0
fase = 0
cant_muestras = 0
frec_muestreo = 0
v_medio = 0

### Señal 1 ###
plt.figure(figsize=(10,10))
tt_1, ss_1 = func_senoidal (a_max = 1, frec = 2000, fase = 0, cant_muestras = 100, frec_muestreo = 30000, v_medio = 0)

plt.subplot (6, 1, 1)
plt.plot (tt_1, ss_1, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.title("Señal de 2kHz")
plt.grid (True)

#SEÑAL 2
tt_2, ss_2 = func_senoidal (a_max = np.pi/2, frec = 2000, fase = np.pi/2, cant_muestras = 100, frec_muestreo = 30000, v_medio = 0)

plt.subplot (6, 1, 2)
plt.plot (tt_2, ss_2, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.title("Señal de 2kHz, desfasada en pi/2")
plt.grid (True)

#SEÑAL 3
a_max = 0
frec = 0
fase = 0
cant_muestras = 0
frec_muestreo = 0
v_medio = 0
tt_3,portadora = func_senoidal (a_max = 1, frec = 1000, fase =0, cant_muestras = 100, frec_muestreo = 30000, v_medio = 0)
tt_3, modulada = func_senoidal (a_max = 1, frec = 2000, fase =0, cant_muestras = 100, frec_muestreo = 30000, v_medio = 0)
final=portadora*modulada

plt.subplot (6, 1, 3)
plt.plot (tt_3, final, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.title("Señal de 2kHz, modulada por 1 kHz")
plt.grid (True)

#SEÑAL 4
a_max = 0
frec = 0
fase = 0
cant_muestras = 0
frec_muestreo = 0
v_medio = 0
ss_clip = np.clip(ss_1, -0.75, 0.75)
tt_1, ss_1 = func_senoidal (a_max = 1, frec = 2000, fase = 0, cant_muestras = 100, frec_muestreo = 30000, v_medio = 0)

plt.subplot (6, 1, 4)
plt.plot (tt_1, ss_clip, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.title("Señal de 2kHz clipeada")
plt.grid (True)

#SEÑAL 5
# Parámetros de la señal
frec = 4000          # frecuencia de la onda cuadrada = 4 kHz
frec_muestreo = 100000  # frecuencia de muestreo (30 kHz, >> 2*frec para cumplir Nyquist)
cant_muestras = 200  # cantidad de muestras
fase = 0
v_medio = 0

# Tiempo
Ts = 1/frec_muestreo
t_final = cant_muestras * Ts
tt = np.arange(0, t_final, Ts)

# Onda cuadrada
onda_cuadrada = sp.square(2 * np.pi * frec * tt + fase)  
plt.subplot (6, 1, 5)
plt.plot (tt, onda_cuadrada, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.title("Señal de 4kHz cuadrada")
plt.grid (True)

#Señal 6


plt.tight_layout()
plt.show()