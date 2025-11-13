import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft

# ============================================================================================
# Función: gráfico de módulo y fase para un filtro digital
# ============================================================================================

def plot_filtro(b, a, w_num, h_num):
    w, h = sig.freqz(b=b, a=a)

    # Fase continua
    fase      = np.unwrap(np.angle(h))
    fase_teo  = np.unwrap(np.angle(h_num))

    # ----------------------------- GRÁFICOS ----------------------------- #

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # Módulo
    ax[0].plot(w, 20*np.log10(np.abs(h)), label="Digital (freqz)")
    ax[0].plot(w_num, 20*np.log10(np.abs(h_num)), '--', color='orange', label="Analítico")
    ax[0].set_title("Respuesta en Módulo")
    ax[0].set_ylabel("|H(ω)| [dB]")
    ax[0].set_xlabel("Pulsación [rad/muestra]")
    ax[0].set_ylim(-50, 20)
    ax[0].grid(True)
    ax[0].legend()

    # Fase
    ax[1].plot(w, np.degrees(fase), label="Digital (freqz)")
    ax[1].plot(w_num, np.degrees(fase_teo), '--', color='orange', label="Analítico")
    ax[1].set_title("Respuesta en Fase")
    ax[1].set_ylabel("Fase [°]")
    ax[1].set_xlabel("Pulsación [rad/muestra]")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================================================
# Coeficientes de los filtros FIR
# ============================================================================================

B_a = [1, 1, 1, 1]
B_b = [1, 1, 1, 1, 1]
B_c = [1, -1]
B_d = [1, 0, -1]

A = 1   # FIR ⇒ denominador igual a 1

# Malla de frecuencias
w = np.linspace(0, np.pi, 1000)

# ============================================================================================
# Expresiones analíticas de cada respuesta en frecuencia
# ============================================================================================

h_a = 2 * np.exp(-1j * 1.5 * w) * (np.cos(1.5*w) + np.cos(0.5*w))
h_b = np.exp(-1j * 2 * w) * (1 + 2*np.cos(w) + 2*np.cos(2*w))
h_c = 2 * np.sin(w/2) * np.exp(1j * (np.pi/2 - w/2))
h_d = 2 * np.sin(w)   * np.exp(1j * (np.pi/2 - w))


# ============================================================================================
# Ejecución
# ============================================================================================

plot_filtro(B_a, A, w, h_a)
plot_filtro(B_b, A, w, h_b)
plot_filtro(B_c, A, w, h_c)
plot_filtro(B_d, A, w, h_d)
