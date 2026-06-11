# TP2 - Parte 3: Registro de escalamiento mediante coordenadas log-polares
# Colocar este archivo en la misma carpeta que las imágenes

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from PIL import Image
import time
import os

OUTDIR = 'Resultados parte 3'
os.makedirs(OUTDIR, exist_ok=True)

def cargar(nombre):
    im  = Image.open(nombre).convert('L')
    arr = np.array(im, dtype=np.float64)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

f1 = cargar('imagen1.jpg')
f5 = cargar('imagen5.jpg')

def mag_espectro(img):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))

def correlacion_fase(a, b):
    A  = np.fft.fft2(a); B = np.fft.fft2(b)
    R  = (A * np.conj(B)) / (np.abs(A * np.conj(B)) + 1e-10)
    c  = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(c), c.shape)
    Ny, Nx = a.shape
    if py > Ny // 2: py -= Ny
    if px > Nx // 2: px -= Nx
    return py, px, c

def cartesiano_a_logpolar(mag, n_radios=256, n_angulos=512):
    Ny, Nx  = mag.shape
    cy, cx  = Ny // 2, Nx // 2
    rho_max = min(Ny, Nx) // 2
    log_rho = np.linspace(0, np.log(rho_max), n_radios)
    rho     = np.exp(log_rho)
    theta   = np.linspace(0, np.pi, n_angulos, endpoint=False)
    R, T    = np.meshgrid(rho, theta, indexing='ij')
    xs = cx + R * np.cos(T)
    ys = cy + R * np.sin(T)
    lp = map_coordinates(mag, [ys, xs], order=1, mode='constant', cval=0)
    return lp, rho_max

# ── Ejecución ─────────────────────────────────────────
N_RAD = 256
m1 = mag_espectro(f1)
m5 = mag_espectro(f5)
lp1, rho_max = cartesiano_a_logpolar(m1, N_RAD)
lp5, _       = cartesiano_a_logpolar(m5, N_RAD)

t0 = time.perf_counter()
drho, _, _ = correlacion_fase(lp1, lp5)
factor = np.exp(drho * np.log(rho_max) / N_RAD)
dt = time.perf_counter() - t0

print("=== Parte 3 — Escalamiento ===")
print(f"Factor de escala estimado: x{factor:.4f}   ({dt*1000:.1f} ms)")

norm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)

# ── Figura 1: imágenes originales ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f1, cmap='gray'); axes[0].set_title('f1 (imagen1)'); axes[0].axis('off')
axes[1].imshow(f5, cmap='gray'); axes[1].set_title('f5 (imagen5, escalada)'); axes[1].axis('off')
plt.suptitle('Parte 3 — Imágenes de entrada')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p3_fig1_imagenes.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 2: espectros de magnitud ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(norm(m1), cmap='inferno'); axes[0].set_title('log|F1(u,v)|'); axes[0].axis('off')
axes[1].imshow(norm(m5), cmap='inferno'); axes[1].set_title('log|F5(u,v)|'); axes[1].axis('off')
plt.suptitle('Parte 3 — Magnitudes espectrales (escala inversa en frecuencia)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p3_fig2_espectros.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 3: mapas log-polares ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(norm(lp1), cmap='plasma', aspect='auto')
axes[0].set_title('Mapa log-polar — |F1|')
axes[0].set_xlabel('eje θ'); axes[0].set_ylabel('eje log(ρ)')
axes[1].imshow(norm(lp5), cmap='plasma', aspect='auto')
axes[1].set_title(f'Mapa log-polar — |F5|  →  factor x{factor:.4f}')
axes[1].set_xlabel('eje θ'); axes[1].set_ylabel('eje log(ρ)')
plt.suptitle('Parte 3 — El cambio de escala se convierte en desplazamiento en log(ρ)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p3_fig3_logpolares.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFiguras guardadas en '{OUTDIR}/':")
print("  p3_fig1_imagenes.png  p3_fig2_espectros.png  p3_fig3_logpolares.png")
