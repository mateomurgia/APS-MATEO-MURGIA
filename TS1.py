import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp


# Función senoidal

def func_senoidal(a_max, frec, fase, cant_muestras, frec_muestreo, v_medio):
    Ts = 1/frec_muestreo
    t_final = cant_muestras * Ts
    tt = np.arange(0, t_final, Ts)
    xx = a_max * np.sin(2 * np.pi * frec * tt + fase) + v_medio
    return tt, xx


# Señales


# Señal 1
frec_muestreo1 = 30000
cant_muestras1 = 100
tt_1, ss_1 = func_senoidal(1, 2000, 0, cant_muestras1, frec_muestreo1, 0)
Ts1 = 1/frec_muestreo1
potencia1 = np.sum(ss_1**2) / cant_muestras1

# Señal 2
tt_2, ss_2 = func_senoidal(np.pi/2, 2000, np.pi/2, 100, 30000, 0)
Ts2 = 1/30000
potencia2 = np.sum(ss_2**2) / len(ss_2)

# Señal 3 (modulación)
tt_3, portadora = func_senoidal(1, 1000, 0, 100, 30000, 0)
_, modulada = func_senoidal(1, 2000, 0, 100, 30000, 0)
final = portadora * modulada
Ts3 = 1/30000
potencia3 = np.sum(final**2) / len(final)

# Señal 4 (clipeada)
tt_4, ss_4 = func_senoidal(1, 2000, 0, 100, 30000, 0)
ss_clip = np.clip(ss_4, -0.75, 0.75)
Ts4 = 1/30000
potencia4 = np.sum(ss_clip**2) / len(ss_clip)

# Señal 5 (cuadrada)
frec5 = 4000
frec_muestreo5 = 100000
cant_muestras5 = 200
Ts5 = 1/frec_muestreo5
t_final5 = cant_muestras5 * Ts5
tt_5 = np.arange(0, t_final5, Ts5)
onda_cuadrada = sp.square(2 * np.pi * frec5 * tt_5)
potencia5 = np.sum(onda_cuadrada**2) / len(onda_cuadrada)

# Señal 6 (pulso rectangular)
frec_muestreo6 = 10000
Ts6 = 1/frec_muestreo6
t_final6 = 0.01
tt_6 = np.arange(0, t_final6, Ts6)
pulso = np.zeros(len(tt_6))
pulso[(tt_6 >= 0.004) & (tt_6 <= 0.006)] = 1.0
energia6 = np.sum(pulso**2) * Ts6


# Productos internos

def producto_interno(x, y):
    return np.sum(x*y)

print("Producto interno Señal 1 - Señal 2:", producto_interno(ss_1, ss_2))
print("Producto interno Señal 1 - Señal 3:", producto_interno(ss_1, final))
print("Producto interno Señal 1 - Señal 4:", producto_interno(ss_1, ss_clip))
print("Producto interno Señal 1 - Señal 5:", producto_interno(ss_1, onda_cuadrada[:len(ss_1)]))
print("Producto interno Señal 1 - Señal 6:", producto_interno(ss_1, pulso[:len(ss_1)]))


# Bloque 1 (plot): Señales

plt.figure(figsize=(10,12))

plt.subplot(6,1,1)
plt.plot(tt_1, ss_1, color='black')
plt.text(0, 1.2, f"Potencia = {potencia1:.3f} J", fontsize=7)
plt.text(0.0005, 1.2, f"#Muestras = {cant_muestras1}", fontsize=7)
plt.text(0.0025, 1.2, f"Ts = {Ts1:.2e} s", fontsize=7)
plt.title("Señal 1: 2 kHz")

plt.subplot(6,1,2)
plt.plot(tt_2, ss_2, color='black')
plt.text(0, 1.85, f"Potencia = {potencia2:.3f} J", fontsize=7)
plt.text(0.0005, 1.85, f"#Muestras = {len(ss_2)}", fontsize=7)
plt.text(0.0025, 1.85, f"Ts = {Ts2:.2e} s", fontsize=7)
plt.title("Señal 2: 2 kHz desfasada π/2")

plt.subplot(6,1,3)
plt.plot(tt_3, final, color='black')
plt.text(0, 0.9, f"Potencia = {potencia3:.3f} J", fontsize=7)
plt.text(0.0005, 0.9, f"#Muestras = {len(final)}", fontsize=7)
plt.text(0.0025, 0.9, f"Ts = {Ts3:.2e} s", fontsize=7)
plt.title("Señal 3: 2 kHz modulada 1 kHz")

plt.subplot(6,1,4)
plt.plot(tt_4, ss_clip, color='black')
plt.text(0, 0.9, f"Potencia = {potencia4:.3f} J", fontsize=7)
plt.text(0.0005, 0.9, f"#Muestras = {len(ss_clip)}", fontsize=7)
plt.text(0.0025, 0.9, f"Ts = {Ts4:.2e} s", fontsize=7)
plt.title("Señal 4: 2 kHz clipeada")

plt.subplot(6,1,5)
plt.plot(tt_5, onda_cuadrada, color='black')
plt.text(0, 1.2, f"Potencia = {potencia5:.3f} J", fontsize=7)
plt.text(0.0005, 1.2, f"#Muestras = {len(onda_cuadrada)}", fontsize=7)
plt.text(0.00150, 1.2, f"Ts = {Ts5:.2e} s", fontsize=7)
plt.title("Señal 5: 4 kHz cuadrada")

plt.subplot(6,1,6)
plt.plot(tt_6, pulso, color='black')
plt.text(0, 1.1, f"Energía = {energia6:.3f} J", fontsize=7)
plt.text(0.002, 1.1, f"#Muestras = {len(tt_6)}", fontsize=7)
plt.text(0.008, 1.1, f"Ts = {Ts6:.2e} s", fontsize=7)
plt.title("Señal 6: Pulso rectangular")

plt.tight_layout()
plt.show()


# Bloque 2(plot): Correlaciones

plt.figure(figsize=(10,12))

plt.subplot(6,1,1)
plt.plot(sp.correlate(ss_1, ss_1, mode='full'), color='black')
plt.title("Autocorrelación Señal 1")

plt.subplot(6,1,2)
plt.plot(sp.correlate(ss_1, ss_2, mode='full'), color='black')
plt.title("Correlación Señal 1 - Señal 2")

plt.subplot(6,1,3)
plt.plot(sp.correlate(ss_1, final, mode='full'), color='black')
plt.title("Correlación Señal 1 - Señal 3")

plt.subplot(6,1,4)
plt.plot(sp.correlate(ss_1, ss_clip, mode='full'), color='black')
plt.title("Correlación Señal 1 - Señal 4")

plt.subplot(6,1,5)
plt.plot(sp.correlate(ss_1, onda_cuadrada[:len(ss_1)], mode='full'), color='black')
plt.title("Correlación Señal 1 - Señal 5")

plt.subplot(6,1,6)
plt.plot(sp.correlate(ss_1, pulso[:len(ss_1)], mode='full'), color='black')
plt.title("Correlación Señal 1 - Señal 6")




# Señal base (2 kHz)

frec_muestreo1 = 30000
cant_muestras1 = 100
frec1 = 2000  # Hz

# alfa = 2000 Hz, beta = 1000 Hz
tt_1, sen_alfa = func_senoidal(1, frec1, 0, cant_muestras1, frec_muestreo1, 0)
tt_2, sen_beta = func_senoidal(1, frec1/2, 0, cant_muestras1, frec_muestreo1, 0)

# Para definir el coseno simplemente desfase el seno pi/2
tt_3, cos_alfa_menos_beta = func_senoidal(1, frec1 - frec1/2, np.pi/2, cant_muestras1, frec_muestreo1, 0)
tt_4, cos_alfa_mas_beta   = func_senoidal(1, frec1 + frec1/2, np.pi/2, cant_muestras1, frec_muestreo1, 0)


identidad = 2*sen_alfa*sen_beta - cos_alfa_menos_beta + cos_alfa_mas_beta

if max(abs(identidad)) < 1e-10:
    print("✅ La identidad se cumple para toda frecuencia !")
else:
    print("❌ La identidad NO se cumple")


plt.tight_layout()

