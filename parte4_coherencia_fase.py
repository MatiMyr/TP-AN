# TP2 - Parte 4: Coherencia de fase y búsqueda del factor k óptimo
# Colocar este archivo en la misma carpeta que las imágenes
# Requiere tifffile: pip install tifffile

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from PIL import Image
import tifffile
import time
import os

OUTDIR = 'Resultados parte 4'
os.makedirs(OUTDIR, exist_ok=True)

def cargar(nombre, ext):
    if ext == 'tif':
        arr = tifffile.imread(f'{nombre}.tif').astype(np.float64)
    else:
        im  = Image.open(f'{nombre}.{ext}').convert('L')
        arr = np.array(im, dtype=np.float64)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

f6_tif = cargar('imagen6', 'tif')
f7_tif = cargar('imagen7', 'tif')
f6_jpg = cargar('imagen6', 'jpg')
f7_jpg = cargar('imagen7', 'jpg')

def reconstruir(mag6, fase7, k):
    k   = max(abs(k), 1e-6)
    img = np.fft.ifft2(mag6 * np.exp(1j * fase7 / k)).real
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-12)

def nitidez(img):
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return float(np.mean(np.hypot(gx, gy)))

def buscar_k(mag6, fase7, k_min=0.05, k_max=2.0, n_grueso=100, n_fino=60):
    ks_g  = np.linspace(k_min, k_max, n_grueso)
    Ss_g  = np.array([nitidez(reconstruir(mag6, fase7, k)) for k in ks_g])
    idx   = int(np.argmax(Ss_g))
    delta = (k_max - k_min) / n_grueso * 2
    ks_f  = np.linspace(max(k_min, ks_g[idx] - delta),
                        min(k_max, ks_g[idx] + delta), n_fino)
    Ss_f  = np.array([nitidez(reconstruir(mag6, fase7, k)) for k in ks_f])
    k_opt = float(ks_f[int(np.argmax(Ss_f))])
    ks_all = np.concatenate([ks_g, ks_f])
    Ss_all = np.concatenate([Ss_g, Ss_f])
    orden  = np.argsort(ks_all)
    return k_opt, ks_all[orden], Ss_all[orden]

# ── Ejecución ─────────────────────────────────────────
print("=== Parte 4 — Coherencia de fase ===")
print("(la búsqueda tarda ~2 minutos, por favor esperar...)\n")

resultados = {}
for label, f6, f7 in [('TIF', f6_tif, f7_tif), ('JPG', f6_jpg, f7_jpg)]:
    mag6  = np.abs(np.fft.fft2(f6))
    fase7 = np.angle(np.fft.fft2(f7))
    t0    = time.perf_counter()
    k_opt, ks, Ss = buscar_k(mag6, fase7)
    dt    = time.perf_counter() - t0
    rec   = reconstruir(mag6, fase7, k_opt)
    resultados[label] = dict(k=k_opt, ks=ks, Ss=Ss, rec=rec)
    print(f"[{label}]  k estimado: {k_opt:.4f}   nitidez: {nitidez(rec):.5f}"
          f"   tiempo: {dt*1000:.0f} ms")

# ── Figura 1: imágenes de entrada TIF ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f6_tif, cmap='gray'); axes[0].set_title('f6 (TIF) — porta magnitud'); axes[0].axis('off')
axes[1].imshow(f7_tif, cmap='gray'); axes[1].set_title('f7 (TIF) — porta fase comprimida'); axes[1].axis('off')
plt.suptitle('Parte 4 — Imágenes de entrada (formato TIF)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig1_entradas_tif.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 2: imágenes de entrada JPG ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f6_jpg, cmap='gray'); axes[0].set_title('f6 (JPG) — porta magnitud'); axes[0].axis('off')
axes[1].imshow(f7_jpg, cmap='gray'); axes[1].set_title('f7 (JPG) — porta fase comprimida'); axes[1].axis('off')
plt.suptitle('Parte 4 — Imágenes de entrada (formato JPG)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig2_entradas_jpg.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 3: curva S(k) TIF ──────────────────────────
r = resultados['TIF']
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r['ks'], r['Ss'], color='green', lw=1.5)
ax.axvline(r['k'], color='gold', lw=2, ls=':', label=f"k óptimo = {r['k']:.4f}")
ax.fill_between(r['ks'], r['Ss'], r['Ss'].min(), color='green', alpha=0.1)
ax.set_title('Parte 4 — Métrica S(k) — TIF (lossless)')
ax.set_xlabel('k'); ax.set_ylabel('S(k) = media |gradiente imagen|')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig3_sk_tif.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 4: imagen recuperada TIF ───────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(r['rec'], cmap='gray')
ax.set_title(f'Imagen recuperada — TIF\nk = {r["k"]:.4f}')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig4_recuperada_tif.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 5: curva S(k) JPG ──────────────────────────
r = resultados['JPG']
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r['ks'], r['Ss'], color='red', lw=1.5)
ax.axvline(r['k'], color='gold', lw=2, ls=':', label=f"k óptimo = {r['k']:.4f}")
ax.fill_between(r['ks'], r['Ss'], r['Ss'].min(), color='red', alpha=0.1)
ax.set_title('Parte 4 — Métrica S(k) — JPG (lossy)')
ax.set_xlabel('k'); ax.set_ylabel('S(k) = media |gradiente imagen|')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig5_sk_jpg.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 6: imagen recuperada JPG ───────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(r['rec'], cmap='gray')
ax.set_title(f'Imagen recuperada — JPG\nk = {r["k"]:.4f}')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig6_recuperada_jpg.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Figura 7: comparación TIF vs JPG ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(resultados['TIF']['rec'], cmap='gray')
axes[0].set_title(f"TIF — k={resultados['TIF']['k']:.4f}"); axes[0].axis('off')
axes[1].imshow(resultados['JPG']['rec'], cmap='gray')
axes[1].set_title(f"JPG — k={resultados['JPG']['k']:.4f}"); axes[1].axis('off')
plt.suptitle('Parte 4 — Comparación imagen recuperada TIF vs JPG')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'p4_fig7_comparacion_tif_jpg.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFiguras guardadas en '{OUTDIR}/':")
print("  p4_fig1_entradas_tif.png     p4_fig2_entradas_jpg.png")
print("  p4_fig3_sk_tif.png           p4_fig4_recuperada_tif.png")
print("  p4_fig5_sk_jpg.png           p4_fig6_recuperada_jpg.png")
print("  p4_fig7_comparacion_tif_jpg.png")
