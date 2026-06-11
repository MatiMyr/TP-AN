# TP2 - Parte 2: Registro de rotación mediante coordenadas polares
# Colocar este archivo en la misma carpeta que las imágenes

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from PIL import Image
import time
import os

OUTDIR = 'Resultados parte 2'
os.makedirs(OUTDIR, exist_ok=True)

def cargar(nombre):
    im  = Image.open(nombre).convert('L')
    arr = np.array(im, dtype=np.float64)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

f3 = cargar('imagen3.jpg')
f4 = cargar('imagen4.jpg')

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

def cartesiano_a_polar(mag, n_angulos):
    Ny, Nx = mag.shape
    cy, cx = Ny // 2, Nx // 2
    nr     = min(Ny, Nx) // 2
    r      = np.linspace(0, nr - 1, nr)
    theta  = np.linspace(0, np.pi, n_angulos, endpoint=False)
    R, T   = np.meshgrid(r, theta, indexing='ij')
    xs = cx + R * np.cos(T)
    ys = cy + R * np.sin(T)
    return map_coordinates(mag, [ys, xs], order=1, mode='constant', cval=0)

# ── Ejecución ─────────────────────────────────────────
N_ANG = 512
m3 = mag_espectro(f3)
m4 = mag_espectro(f4)
p3 = cartesiano_a_polar(m3, N_ANG)
p4 = cartesiano_a_polar(m4, N_ANG)

t0 = time.perf_counter()
_, dtheta, _ = correlacion_fase(p3, p4)
angulo = dtheta * 180.0 / N_ANG
dt = time.perf_counter() - t0

print("=== Parte 2 — Rotación ===")
print(f"Ángulo estimado: {angulo:.2f}°   ({dt*1000:.1f} ms, N_ANG = {N_ANG})")

norm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)


# ── Figura 1: imágenes originales ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f3, cmap='gray'); axes[0].set_title('f3 (imagen3)'); axes[0].axis('off')
axes[1].imshow(f4, cmap='gray'); axes[1].set_title('f4 (imagen4, rotada)'); axes[1].axis('off')
plt.suptitle('Parte 2 — Imágenes de entrada')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p2_fig1_imagenes.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 2: espectros de magnitud ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(norm(m3), cmap='inferno'); axes[0].set_title('log|F3(u,v)|'); axes[0].axis('off')
axes[1].imshow(norm(m4), cmap='inferno'); axes[1].set_title('log|F4(u,v)|'); axes[1].axis('off')
plt.suptitle('Parte 2 — Magnitudes espectrales (ambas rotadas igual)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p2_fig2_espectros.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 3: mapas polares ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(norm(p3), cmap='plasma', aspect='auto')
axes[0].set_title('Mapa polar — |F3|')
axes[0].set_xlabel('eje θ (0° → 180°)'); axes[0].set_ylabel('eje ρ (radio)')
axes[1].imshow(norm(p4), cmap='plasma', aspect='auto')
axes[1].set_title(f'Mapa polar — |F4|  →  Δθ = {angulo:.2f}°')
axes[1].set_xlabel('eje θ (0° → 180°)'); axes[1].set_ylabel('eje ρ (radio)')
plt.suptitle('Parte 2 — La rotación se convierte en desplazamiento en θ')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p2_fig3_polares.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFiguras guardadas en '{OUTDIR}/':")
print("  p2_fig1_imagenes.png  p2_fig2_espectros.png  p2_fig3_polares.png")
