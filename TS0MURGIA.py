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

a_max = 0
frec = 0
fase = 0
cant_muestras = 0
frec_muestreo = 0
v_medio = 0

### Señal 1 ###
plt.figure(figsize=(8,8))
tt_1, ss_1 = func_senoidal (a_max = 1, frec = 10, fase = 0, cant_muestras = 1000, frec_muestreo = 1000, v_medio = 0)

plt.subplot (6, 1, 1)
plt.plot (tt_1, ss_1, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.grid (True)

### Señal 2 ###

tt_2, ss_2 = func_senoidal (a_max = 1, frec = 500, fase = 0, cant_muestras = 1000, frec_muestreo = 1000, v_medio = 0)

plt.subplot (6, 1, 2)
plt.plot (tt_2, ss_2, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.grid (True)

### Señal 3 ###

tt_3, ss_3 = func_senoidal (a_max = 1, frec = 999, fase = 0, cant_muestras = 1000, frec_muestreo = 1000, v_medio = 0)

plt.subplot (6, 1, 3)
plt.plot (tt_3, ss_3, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.grid (True)

### Señal 4 ###

tt_4, ss_4 = func_senoidal (a_max = 1, frec = 1001, fase = 0, cant_muestras = 1000, frec_muestreo = 1000, v_medio = 0)

plt.subplot (6, 1, 4)
plt.plot (tt_4, ss_4, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.grid (True)

### Señal 5 ###

tt_5, ss_5 = func_senoidal (a_max = 1, frec = 2001, fase = 0, cant_muestras = 1000, frec_muestreo = 1000, v_medio = 0)

plt.subplot (6, 1, 5)
plt.plot (tt_5, ss_5, linestyle='-', color='black')
plt.xlabel ("Tiempo")
plt.ylabel ("Volts")
plt.grid (True)


plt.tight_layout()
plt.show ()