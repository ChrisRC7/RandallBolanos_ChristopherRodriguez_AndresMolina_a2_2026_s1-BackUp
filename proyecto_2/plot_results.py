"""
plot_results.py — Genera gráficas de rendimiento para DLP Ray Tracing
CE 4302 Arquitectura de Computadores II

Lee los CSVs de resultados/tablas/ y guarda PNGs en resultados/graficas/.
Ejecutar desde proyecto_2/:
    python plot_results.py
"""

import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("ERROR: matplotlib no encontrado.")
    print("Instalar con:  pip install matplotlib")
    sys.exit(1)

TABLAS_DIR   = os.path.join("resultados", "tablas")
GRAFICAS_DIR = os.path.join("resultados", "graficas")

os.makedirs(GRAFICAS_DIR, exist_ok=True)

# Colores por implementación
C_SEQ  = "#4c72b0"
C_SIMD = "#dd8452"
C_CUDA = "#55a868"
C_IDEAL = "#808080"


def _load_csv(filename):
    path = os.path.join(TABLAS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {path}\n"
            "Ejecuta primero: mingw32-make bench"
        )
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Gráfica 1 — Tiempo de ejecución vs. resolución (escala log)
# ---------------------------------------------------------------------------
def plot_tiempo_vs_resolucion():
    rows = _load_csv("escalabilidad.csv")

    labels  = [r["Resolucion"]      for r in rows]
    t_seq   = [float(r["T_SEQ_ms"]) for r in rows]
    t_simd  = [float(r["T_SIMD_ms"])for r in rows]
    t_cuda  = [float(r["T_CUDA_ms"])for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, t_seq,  "o-", color=C_SEQ,  lw=2, ms=7, label="SEQ  (Phong + sombras)")
    ax.plot(x, t_simd, "s-", color=C_SIMD, lw=2, ms=7, label="SIMD (AVX2, intersección)")
    ax.plot(x, t_cuda, "^-", color=C_CUDA, lw=2, ms=7, label="CUDA (GPU, intersección)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Tiempo de ejecución (ms) — escala log")
    ax.set_title("Tiempo de ejecución vs. Tamaño del problema")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    fig.tight_layout()

    out = os.path.join(GRAFICAS_DIR, "tiempo_vs_resolucion.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Gráfica 2 — Speedup vs. SEQ a 1600×900
# ---------------------------------------------------------------------------
def plot_speedup():
    rows     = _load_csv("speedup_eficiencia.csv")
    impls    = [r["Impl"]           for r in rows]
    speedups = [float(r["Speedup"]) for r in rows]
    colors   = [C_SEQ, C_SIMD, C_CUDA]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(impls, speedups, color=colors, width=0.5, edgecolor="white", zorder=3)

    for bar, val in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(speedups) * 0.02,
            f"{val:.2f}×",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.axhline(y=1, color=C_IDEAL, lw=1.2, linestyle="--", label="Baseline (1×)")
    ax.set_ylabel("Speedup vs. SEQ")
    ax.set_title("Speedup a 1600×900")
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, zorder=0)
    fig.tight_layout()

    out = os.path.join(GRAFICAS_DIR, "speedup.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Gráfica 3 — Eficiencia paralela a 1600×900
# ---------------------------------------------------------------------------
def plot_eficiencia():
    rows   = _load_csv("speedup_eficiencia.csv")
    impls  = [r["Impl"]                for r in rows]
    effs   = [float(r["Eficiencia_pct"]) for r in rows]
    colors = [C_SEQ, C_SIMD, C_CUDA]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(impls, effs, color=colors, width=0.5, edgecolor="white", zorder=3)

    for bar, val in zip(bars, effs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(effs) * 0.02,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.axhline(y=100, color="crimson", lw=1.5, linestyle="--", label="Ideal (100%)")
    ax.set_ylabel("Eficiencia paralela (%)")
    ax.set_title("Eficiencia paralela a 1600×900")
    ax.set_ylim(0, max(effs) * 1.25)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, zorder=0)
    fig.tight_layout()

    out = os.path.join(GRAFICAS_DIR, "eficiencia_paralela.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Gráfica 4 — Escalabilidad normalizada vs. ideal O(W·H)
# ---------------------------------------------------------------------------
def plot_escalabilidad_normalizada():
    rows    = _load_csv("escalabilidad_normalizada.csv")
    labels  = [r["Resolucion"]          for r in rows]
    factors = [float(r["Factor_pix"])   for r in rows]
    r_seq   = [float(r["T_SEQ_rel"])    for r in rows]
    r_simd  = [float(r["T_SIMD_rel"])   for r in rows]
    r_cuda  = [float(r["T_CUDA_rel"])   for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, factors, "k--",  lw=1.5,               label="Ideal O(W·H)")
    ax.plot(x, r_seq,   "o-",   color=C_SEQ,  lw=2, ms=7, label="SEQ")
    ax.plot(x, r_simd,  "s-",   color=C_SIMD, lw=2, ms=7, label="SIMD")
    ax.plot(x, r_cuda,  "^-",   color=C_CUDA, lw=2, ms=7, label="CUDA")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tiempo relativo (normalizado a 400×225 = 1×)")
    ax.set_title("Escalabilidad normalizada\n(ideal: duplicar resolución → ×4 el tiempo)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()

    out = os.path.join(GRAFICAS_DIR, "escalabilidad_normalizada.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Eficiencia vs. ideal — identifica el punto aproximado de cuello de botella
# ---------------------------------------------------------------------------
def plot_eficiencia_ideal():
    rows    = _load_csv("escalabilidad_normalizada.csv")
    labels  = [r["Resolucion"] for r in rows]
    factors = [float(r["Factor_pix"]) for r in rows]
    r_seq   = [float(r["T_SEQ_rel"]) for r in rows]
    r_simd  = [float(r["T_SIMD_rel"]) for r in rows]
    r_cuda  = [float(r["T_CUDA_rel"]) for r in rows]
    x = np.arange(len(labels))

    def efficiency(rel_times):
        return [f / t if t > 0 else 0.0 for f, t in zip(factors, rel_times)]

    eff_seq  = efficiency(r_seq)
    eff_simd = efficiency(r_simd)
    eff_cuda = efficiency(r_cuda)

    threshold = 0.80

    def first_below(values):
        for i, v in enumerate(values):
            if v < threshold:
                return i
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, eff_seq,  "o-", color=C_SEQ,  lw=2, ms=7, label="SEQ")
    ax.plot(x, eff_simd, "s-", color=C_SIMD, lw=2, ms=7, label="SIMD")
    ax.plot(x, eff_cuda, "^-", color=C_CUDA, lw=2, ms=7, label="CUDA")
    ax.axhline(threshold, color=C_IDEAL, lw=1.5, linestyle="--",
               label="Umbral de cuello de botella (80% del ideal)")

    for name, vals, color in [("SEQ", eff_seq, C_SEQ), ("SIMD", eff_simd, C_SIMD), ("CUDA", eff_cuda, C_CUDA)]:
        idx = first_below(vals)
        if idx is not None:
            ax.scatter([x[idx]], [vals[idx]], color=color, s=90, zorder=5, edgecolor="black")
            ax.annotate(
                f"cuello de botella\n≈ {labels[idx]}",
                xy=(x[idx], vals[idx]),
                xytext=(x[idx] + 0.1, vals[idx] + 0.12),
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.2, max(max(eff_seq), max(eff_simd), max(eff_cuda)) * 1.15))
    ax.set_ylabel("Eficiencia vs. ideal (1.0 = escala lineal perfecta)")
    ax.set_title("Aparición aproximada del cuello de botella\n(eficiencia respecto al ideal)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(GRAFICAS_DIR, "eficiencia_vs_ideal.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Boxplot — distribución de tiempos por configuración (impl × resolución)
# ---------------------------------------------------------------------------
def plot_boxplot_distribuciones():
    rows = _load_csv("distribuciones.csv")

    resols = ["400x225", "800x450", "1600x900", "3200x1800"]
    impls  = [("SEQ", C_SEQ), ("SIMD", C_SIMD), ("CUDA", C_CUDA)]

    data = {}
    for r in rows:
        key = (r["Impl"], r["Resolucion"])
        if key not in data:
            data[key] = []
        t = float(r["T_ms"])
        if t > 0:
            data[key].append(t)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (impl, color) in zip(axes, impls):
        box_data = [data.get((impl, res), [0]) for res in resols]
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels(resols, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Tiempo (ms)")
        ax.set_title(impl)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)

    fig.suptitle("Distribución de tiempos por configuración (boxplot)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(GRAFICAS_DIR, "boxplot_distribuciones.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Histograma de frecuencia — distribución en 1600×900
# ---------------------------------------------------------------------------
def plot_histograma_distribuciones():
    rows = _load_csv("distribuciones.csv")

    data = {"SEQ": [], "SIMD": [], "CUDA": []}
    for r in rows:
        if r["Resolucion"] == "1600x900":
            t = float(r["T_ms"])
            if t > 0 and r["Impl"] in data:
                data[r["Impl"]].append(t)

    impls = [("SEQ", C_SEQ), ("SIMD", C_SIMD), ("CUDA", C_CUDA)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (impl, color) in zip(axes, impls):
        times = data[impl]
        if not times:
            continue
        n    = len(times)
        bins = min(10, max(5, n // 2))
        ax.hist(times, bins=bins, color=color, edgecolor="white", alpha=0.85, zorder=3)
        mean = np.mean(times)
        std  = np.std(times, ddof=1) if n > 1 else 0.0
        ci   = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
        ax.axvline(mean,      color="black",   lw=1.5, linestyle="--",
                   label=f"Media: {mean:.2f} ms")
        ax.axvline(mean - ci, color="crimson", lw=1.2, linestyle=":",
                   label=f"IC 95%: ±{ci:.2f}")
        ax.axvline(mean + ci, color="crimson", lw=1.2, linestyle=":")
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"{impl}  (n={n})\nσ = {std:.3f} ms")
        ax.legend(fontsize=7.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45, zorder=0)

    fig.suptitle("Histograma de tiempos a 1600×900", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(GRAFICAS_DIR, "histograma_distribuciones.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Intervalos de confianza 95% — barras de error por configuración
# ---------------------------------------------------------------------------
def plot_intervalos_confianza():
    rows = _load_csv("estadisticas.csv")

    resols     = ["400x225", "800x450", "1600x900", "3200x1800"]
    impls_info = [("SEQ", C_SEQ), ("SIMD", C_SIMD), ("CUDA", C_CUDA)]

    stats = {}
    for r in rows:
        stats[(r["Impl"], r["Resolucion"])] = (
            float(r["Media_ms"]),
            float(r["CI95_low"]),
            float(r["CI95_high"]),
        )

    x     = np.arange(len(resols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (impl, color) in enumerate(impls_info):
        means  = [stats[(impl, r)][0] for r in resols]
        err_lo = [max(0.0, stats[(impl, r)][0] - stats[(impl, r)][1]) for r in resols]
        err_hi = [max(0.0, stats[(impl, r)][2] - stats[(impl, r)][0]) for r in resols]

        offset = (i - 1) * width
        ax.bar(x + offset, means, width, color=color, alpha=0.82,
               label=impl, edgecolor="white", zorder=3)
        ax.errorbar(x + offset, means,
                    yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(resols)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Tiempo medio (ms) ± IC 95%  —  escala log")
    ax.set_title("Intervalos de confianza 95% por configuración")
    ax.legend()
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.45, zorder=0)
    fig.tight_layout()

    out = os.path.join(GRAFICAS_DIR, "intervalos_confianza.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generando gráficas de rendimiento...")
    plot_tiempo_vs_resolucion()
    plot_speedup()
    plot_eficiencia()
    plot_escalabilidad_normalizada()
    plot_eficiencia_ideal()
    plot_boxplot_distribuciones()
    plot_histograma_distribuciones()
    plot_intervalos_confianza()
    print(f"Gráficas guardadas en: {GRAFICAS_DIR}\\")
