import matplotlib.pyplot as plt
import numpy as np

def func_senoidal(a_max, frec, fase, cant_muestras, frec_muestreo, v_medio):
    Ts = 1/frec_muestreo
    t_final = cant_muestras * Ts
    tt = np.arange(0, t_final, Ts)
    xx = a_max * np.sin(2*np.pi*frec*tt + fase) + v_medio
    return tt, xx

plt.figure(figsize=(8, 8))   # <<---  ancho x alto en pulgadas

# ---- Señal 1 ----
tt_1, ss_1 = func_senoidal(1, 10, 0, 1000, 1000, 0)
plt.subplot(5, 1, 1)
plt.plot(tt_1, ss_1, color="black")
plt.title("Señal 1 - 10 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Volts")
plt.grid(True)

# ---- Señal 2 ----
tt_2, ss_2 = func_senoidal(1, 500, 0, 1000, 1000, 0)
plt.subplot(5, 1, 2)
plt.plot(tt_2, ss_2, color="black")
plt.title("Señal 2 - 500 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Volts")
plt.grid(True)

# ---- Señal 3 ----
tt_3, ss_3 = func_senoidal(1, 999, 0, 1000, 1000, 0)
plt.subplot(5, 1, 3)
plt.plot(tt_3, ss_3, color="black")
plt.title("Señal 3 - 999 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Volts")
plt.grid(True)

# ---- Señal 4 ----
tt_4, ss_4 = func_senoidal(1, 1001, 0, 1000, 1000, 0)
plt.subplot(5, 1, 4)
plt.plot(tt_4, ss_4, color="black")
plt.title("Señal 4 - 1001 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Volts")
plt.grid(True)

# ---- Señal 5 ----
tt_5, ss_5 = func_senoidal(1, 2001, 0, 1000, 1000, 0)
plt.subplot(5, 1, 5)
plt.plot(tt_5, ss_5, color="black")
plt.title("Señal 5 - 2001 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Volts")
plt.grid(True)

plt.tight_layout()
plt.show()
