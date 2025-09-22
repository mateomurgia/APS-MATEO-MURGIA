import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal.windows as window

# ---------------- Parámetros ----------------
fs = 1000
N  = 1000
df = fs/N
R  = 200
Nfft = 10*N

def func_senoidal(tt, frec, amp, fase=0, v_medio=0):
    return amp*np.sin(2*np.pi*frec*tt + fase) + v_medio

# --- Mallas tiempo y frecuencias por columna ---
tt = np.arange(N)/fs
tt_col = tt.reshape(N,1)                         # (N,1)
fr_vec = np.random.uniform(-2, 2, R)             # (R,)
f0_vec = (N/4 + fr_vec) * df                     # (R,) Hz
f0_row = f0_vec.reshape(1,R)                     # (1,R)
TT = np.tile(tt_col, (1,R))                      # (N,R)
F0 = np.tile(f0_row, (N,1))                      # (N,R)

# --- Señal + ruido (SNR en dB) ---
amp_0 = np.sqrt(2)
SNR = 3
s_1 = func_senoidal(TT, F0, amp_0)               # (N,R)

P_signal = amp_0**2 / 2
P_noise  = P_signal / (10**(SNR/10))
ruido_mat = np.random.normal(0, np.sqrt(P_noise), size=(N,R))
x_1 = s_1 + ruido_mat

print(f"Var ruido target={P_noise:.4f}  empírica≈{np.var(ruido_mat):.4f}")

# --- Ventanas ---
w_rect = np.ones((N,1))
w_flat = window.flattop(N, sym=False).reshape(-1,1)
w_bh   = window.blackmanharris(N, sym=False).reshape(-1,1)
w_hann = window.hann(N, sym=False).reshape(-1,1)

ventanas = [
    ("rectangular",    w_rect),
    ("flattop",        w_flat),
    ("blackmanharris", w_bh),
    ("hann",           w_hann),
]

# --- Semieje positivo con kpos ---
kpos = slice(0, Nfft//2 + 1)
f_pos = np.arange(Nfft)[kpos] * (fs/Nfft)

# --- Un gráfico POR ventana, con TODOS los espectros (todas las columnas) ---
for nombre, w in ventanas:
    X = fft(x_1 * w, n=Nfft, axis=0) / N          # (Nfft,R)
    P = np.abs(X)**2                               # (Nfft,R) potencia por bin
    Pp_cols = P[kpos, :]                           # (Nfft/2+1, R)
    Pp_cols = np.maximum(Pp_cols, np.finfo(float).tiny)
    Pdb_cols = 10*np.log10(Pp_cols)

    plt.figure()
    # dibuja las R curvas; líneas finas y algo de transparencia para que no tape
    plt.plot(f_pos, Pdb_cols, linewidth=0.6, alpha=0.6)
    plt.title(f"Espectros – ventana {nombre}")
    plt.xlim(0, fs/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Potencia [dB]  (|FFT|^2 por realización)")
    plt.grid(True, ls=':')
    plt.tight_layout()

plt.show()
