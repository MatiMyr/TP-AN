# TP2 - Parte 1: Estimación de traslación mediante correlación de fase
# Colocar este archivo en la misma carpeta que las imágenes

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from PIL import Image
import time
import os

OUTDIR = 'Resultados parte 1'
os.makedirs(OUTDIR, exist_ok=True)

def cargar(nombre):
    im  = Image.open(nombre).convert('L')
    arr = np.array(im, dtype=np.float64)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

f1 = cargar('imagen1.jpg')
f2 = cargar('imagen2.jpg')

def correlacion_fase(a, b):
    A  = np.fft.fft2(a)
    B  = np.fft.fft2(b)
    R  = (A * np.conj(B)) / (np.abs(A * np.conj(B)) + 1e-10)
    c  = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(c), c.shape)
    Ny, Nx = a.shape
    if py > Ny // 2: py -= Ny
    if px > Nx // 2: px -= Nx
    return py, px, c

def metodo_espacial(a, b, rango=50):
    mejor_mse = np.inf
    mejor_d   = (0, 0)
    for dy in range(-rango, rango + 1):
        for dx in range(-rango, rango + 1):
            b_mov = shift(b, shift=(-dy, -dx), mode='wrap')
            mse   = np.mean((a - b_mov) ** 2)
            if mse < mejor_mse:
                mejor_mse = mse
                mejor_d   = (dy, dx)
    return mejor_d

# ── Ejecución ─────────────────────────────────────────
print("=== Parte 1 — Traslación ===")

t0 = time.perf_counter()
dy, dx, corr = correlacion_fase(f1, f2)
t_fft = time.perf_counter() - t0
print(f"[FFT]      dy={dy:4d}, dx={dx:4d}   →  {t_fft*1000:.1f} ms")

t0 = time.perf_counter()
d_esp = metodo_espacial(f1, f2, rango=50)
t_esp = time.perf_counter() - t0
print(f"[Espacial] dy={d_esp[0]:4d}, dx={d_esp[1]:4d}   →  {t_esp*1000:.1f} ms")
print(f"Speedup FFT vs Espacial: x{t_esp/t_fft:.0f}")

norm   = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)
corr_s = np.fft.fftshift(corr)
cy, cx = np.array(corr_s.shape) // 2

# ── Figura 1: imágenes originales ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f1, cmap='gray'); axes[0].set_title('f1 (imagen1)'); axes[0].axis('off')
axes[1].imshow(f2, cmap='gray'); axes[1].set_title('f2 (imagen2)'); axes[1].axis('off')
plt.suptitle('Parte 1 — Imágenes de entrada')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p1_fig1_imagenes.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 2: diferencia entre imágenes ───────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.abs(f1 - f2), cmap='hot')
ax.set_title('Diferencia absoluta |f1 - f2|')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p1_fig2_diferencia.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 3: espectros de magnitud ───────────────────
mag1 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f1))))
mag2 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f2))))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(norm(mag1), cmap='inferno'); axes[0].set_title('log|F1(u,v)|'); axes[0].axis('off')
axes[1].imshow(norm(mag2), cmap='inferno'); axes[1].set_title('log|F2(u,v)|'); axes[1].axis('off')
plt.suptitle('Parte 1 — Magnitudes espectrales (invariantes a traslación)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p1_fig3_espectros.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 4: correlación de fase con pico ────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(norm(corr_s), cmap='plasma')
ax.plot(cx + dx, cy + dy, 'r+', ms=16, mew=2.5)
ax.set_title(f'Correlación de fase\nPico: dy={dy}, dx={dx}')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p1_fig4_correlacion.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFiguras guardadas en '{OUTDIR}/':")
print("  p1_fig1_imagenes.png  p1_fig2_diferencia.png")
print("  p1_fig3_espectros.png  p1_fig4_correlacion.png")
