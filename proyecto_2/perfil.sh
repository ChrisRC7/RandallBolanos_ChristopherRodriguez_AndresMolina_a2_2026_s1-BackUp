#!/usr/bin/env bash
# =============================================================================
# perfil.sh — Perfilado de desempeño: SEQ, SIMD y CUDA
# CE 4302 Arquitectura de Computadores II
#
# Hardware objetivo:
#   CPU: AMD Ryzen 7 7700 (Zen 4, family 25h, model 61h)
#   GPU: NVIDIA GeForce RTX 4060 (SM 8.9, 24 SMs)
#
# Herramientas:
#   perf  — Linux Performance Events (SEQ y SIMD)
#   ncu   — Nsight Compute CLI (kernel CUDA)
#   nsys  — Nsight Systems CLI (timeline CPU+GPU con PCIe)
#
# Uso:
#   chmod +x perfil.sh
#   sudo ./perfil.sh          (sudo necesario para ncu y algunos eventos perf)
#
# Salidas:
#   resultados/perfilado/perf_seq.txt
#   resultados/perfilado/perf_simd.txt
#   resultados/perfilado/ncu_cuda.txt
#   resultados/perfilado/nsys_cuda.nsys-rep  (abrir con Nsight Systems GUI)
# =============================================================================

set -e
OUTDIR="resultados/perfilado"
mkdir -p "$OUTDIR"

W=1600; H=900

echo "================================================================"
echo " PERFILADO DLP Ray Tracing — ${W}x${H}"
echo " $(date)"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — SEQ: perf stat
#
# Métricas elegidas y justificación:
#
#  cycles, instructions → IPC (instrucciones por ciclo).
#    Justificación: el SEQ es el baseline escalar. IPC alto (~2.8) confirma
#    que el pipeline Zen 4 está bien utilizado con -O2. Sirve para comparar
#    contra SIMD donde esperamos IPC mayor (más trabajo por ciclo).
#
#  r0803 (FpRetSseAvxOps, umask=0x08, AMD Zen 4) → ops FP 256-bit packed single.
#    Justificación: mide cuántas instrucciones AVX2 YMM de 256 bits emite el
#    compilador incluso en código "escalar" (auto-vectorización de GCC -O2).
#    Permite cuantificar la diferencia real con la versión SIMD manual.
#
#  r0403 (umask=0x04) → ops FP 128-bit packed single (SSE/AVX 128).
#    Justificación: GCC -O2 puede emitir XMM ops. Necesario para entender
#    la mezcla vectorial del SEQ y diferenciarlo del SIMD.
#
#  r0103 (umask=0x01) → ops FP escalares single.
#    Justificación: confirma que el SEQ ejecuta principalmente ops escalares,
#    validando que SIMD realmente logra paralelismo a nivel de datos.
#
#  L1-dcache-loads, L1-dcache-load-misses → tasa de fallos L1.
#    Justificación: el ray tracer accede a la escena (esferas) repetidamente.
#    Una tasa de fallos L1 baja confirma que la escena cabe en caché L1/L2.
#    Alta tasa de fallos sugeriría que escalar a más esferas penaliza latencia.
#
#  cache-misses → fallos en LLC (Last Level Cache, L3).
#    Justificación: fallos LLC obligan a ir a DRAM (~50-100 ns). En el SEQ
#    son bajos porque la escena (~5 esferas, ~100 bytes) cabe en L1.
#
#  branch-misses → predicción de ramas.
#    Justificación: ray tracing tiene ramas divergentes (hit/miss por esfera,
#    prueba de sombra, decisión de reflexión). Las mispredictions generan
#    stalls de ~15 ciclos en el pipeline. Con 0.33% de miss-rate en SEQ, el
#    predictor Zen 4 maneja bien las ramas del bucle interno.
#
#  stalled-cycles-frontend → ciclos de stall en el front-end.
#    Justificación: si el front-end del pipeline no puede decodificar
#    instrucciones a tiempo (fetch/decode bottleneck), limita el IPC.
#    SIMD tiene ~28% de stall-frontend porque los registros YMM tienen mayor
#    latencia de decodificación en Zen 4.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── [1/3] perf stat SEQ ─────────────────────────────────────────"
perf stat -r 5 \
  -e cycles,instructions \
  -e r0103,r0403,r0803,rff03 \
  -e L1-dcache-loads,L1-dcache-load-misses \
  -e cache-references,cache-misses \
  -e branches,branch-misses \
  -e stalled-cycles-frontend \
  ./bin/renderer_seq $W $H /dev/null \
  2>&1 | tee "$OUTDIR/perf_seq.txt"

echo ""
echo "── [2/3] perf stat SIMD ────────────────────────────────────────"
# Métricas idénticas a SEQ para comparación directa.
# Diferencias esperadas vs SEQ:
#   r0803 (+6.8×): confirma que _mm256_fmadd_ps emite instrucciones YMM reales.
#   r0103 (−5×):   menos ops escalares porque el bucle procesa 8px a la vez.
#   branch-misses (~igual): el predictor no puede predecir el residuo escalar.
#   stalled-cycles-frontend (+27%): decode de YMM más lento que XMM/scalar.
#   IPC (~2.1 vs 2.8): YMM usa más recursos de ejecución → dispatch más lento.
perf stat -r 5 \
  -e cycles,instructions \
  -e r0103,r0403,r0803,rff03 \
  -e L1-dcache-loads,L1-dcache-load-misses \
  -e cache-references,cache-misses \
  -e branches,branch-misses \
  -e stalled-cycles-frontend \
  ./bin/renderer_simd_dummy $W $H /dev/null \
  2>&1 | tee "$OUTDIR/perf_simd.txt"

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — CUDA: ncu (Nsight Compute CLI)
#
# Métricas elegidas y justificación:
#
#  Compute (SM) Throughput (88.4%) → utilización del SM.
#    Justificación: mide qué fracción del tiempo los SMs están ejecutando
#    instrucciones. 88% indica que el kernel está compute-bound, no
#    memory-bound: la escena pequeña en __constant__ evita esperas de memoria.
#
#  Achieved Occupancy (88.2%) vs Theoretical (100%) → ocupación de GPU.
#    Justificación: la ocupación mide cuántos warps están activos
#    simultáneamente sobre el SM. Bloques 16×16 = 256 threads = 8 warps.
#    88% ≈ 42 warps / 48 máx por SM (Zen 4 / Ada Lovelace). Alta ocupación
#    → el SM puede ocultar latencias de memoria con otros warps activos.
#
#  Avg. Active Threads Per Warp (31.95 / 32) → divergencia de warp.
#    Justificación: un warp tiene 32 hilos. Si algunos toman ramas distintas
#    (ej. hit vs miss de esfera), los inactivos desperdician ciclos.
#    31.95 ≈ 32 indica divergencia casi nula: todos los rayos del warp
#    generalmente siguen el mismo camino (hit o miss de todas las esferas).
#
#  FMA pipeline utilization (47.3%) → unidades de punto flotante.
#    Justificación: confirma que el cuello de botella es aritmético FP
#    (multiplicaciones + sumas en intersección rayo-esfera), no accesos a
#    memoria. Justifica que implementar shading Phong completo solo aumentará
#    tiempo de compute, no requerirá más ancho de banda.
#
#  DRAM Throughput (90.3% del pico = 238 GB/s) → ancho de banda DRAM.
#    Justificación: el framebuffer SoA (3 arrays R/G/B separados) maximiza
#    la coalescencia de escrituras. Un framebuffer AoS (struct rgb) generaría
#    escrituras no coalescentes → DRAM Throughput más bajo. Este dato valida
#    la decisión de diseño SoA.
#
#  L2 Hit Rate (97.96%) → efectividad de la memoria __constant__.
#    Justificación: la escena en __constant__ se cachea en L2. 98% de hit
#    significa que prácticamente todas las lecturas de esferas y cámara se
#    sirven desde L2 sin ir a DRAM → confirma que __constant__ fue la
#    elección correcta para datos de solo lectura broadcast.
#
#  L1/TEX Hit Rate (0.81%) → framebuffer en L1.
#    Justificación: las escrituras al framebuffer van directo a L2/DRAM
#    (write-invalidate en L1). 0.81% es normal y esperado para escrituras
#    de un solo resultado por hilo (no hay reutilización).
#
#  Issue Slots Busy (90.07%) → eficiencia del planificador de warps.
#    Justificación: 90% de los ciclos el planificador emite instrucciones.
#    10% de "no eligible warps" sugiere pequeñas esperas de dependencias
#    de datos (latencia de FMA ~4 ciclos en Ada Lovelace).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── [3/3] ncu CUDA (perfilado) ──────────────────────────────────"
ncu -f --set full \
    --target-processes all \
    --export "$OUTDIR/ncu_cuda_report" \
    ./bin/renderer_cuda_dummy $W $H /dev/null \
    2>&1 | tee "$OUTDIR/ncu_cuda.txt"

echo ""
echo "── [3b/3] ncu CUDA (importar métricas legibles) ────────────────"
ncu --import "$OUTDIR/ncu_cuda_report.ncu-rep" --page raw \
  2>&1 | tee "$OUTDIR/ncu_cuda_full.txt"

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — CUDA: nsys (Nsight Systems) — timeline CPU + GPU + PCIe
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — CUDA: barrido multi-resolución (bitácora de cuellos de botella)
#
# Ejecuta ncu con 5 métricas clave en varias resoluciones para observar a partir
# de qué tamaño de imagen cada métrica cruza los umbrales de 80% / 50%.
# Solo se recolectan las métricas necesarias (--metrics) para minimizar overhead.
#
# Resoluciones del barrido:
#   256×144   → ~37k píxeles   (carga mínima)
#   512×288   → ~147k píxeles
#   800×450   → ~360k píxeles
#   1280×720  → ~922k píxeles  (HD)
#   1600×900  → ~1.44M píxeles (perfil principal)
#   1920×1080 → ~2.07M píxeles (Full HD)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── [4/5] Barrido multi-resolución CUDA (bitácora) ──────────────"
SWEEP_DIR="$OUTDIR/sweep"
mkdir -p "$SWEEP_DIR"

SWEEP_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed"
SWEEP_METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"
SWEEP_METRICS+=",smsp__issue_active.avg.pct_of_peak_sustained_active"
SWEEP_METRICS+=",gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
SWEEP_METRICS+=",sm__inst_issued.avg.per_cycle_active"

for RES in "256x144" "512x288" "800x450" "1280x720" "1600x900" "1920x1080"; do
    RW="${RES%x*}"
    RH="${RES#*x}"
    echo "  Midiendo resolución ${RES} (${RW}×${RH})..."
    ncu -f \
        --metrics "$SWEEP_METRICS" \
        --target-processes all \
        --export "$SWEEP_DIR/ncu_sweep_${RES}" \
        ./bin/renderer_cuda_dummy "$RW" "$RH" /dev/null \
        > /dev/null 2>&1
    ncu --import "$SWEEP_DIR/ncu_sweep_${RES}.ncu-rep" --page raw \
        > "$SWEEP_DIR/ncu_sweep_${RES}.txt" 2>&1
    echo "    -> $SWEEP_DIR/ncu_sweep_${RES}.txt"
done

echo ""
echo "── [5/5] nsys CUDA timeline ────────────────────────────────────"
nsys profile \
    --stats=true \
    --output="$OUTDIR/nsys_cuda" \
    --force-overwrite=true \
    ./bin/renderer_cuda_dummy $W $H /dev/null \
    2>&1 | tee "$OUTDIR/nsys_cuda_stats.txt"

echo ""
echo "── [5/5] Generando CSVs y gráficas de perfilado ────────────────"
# Devolver propiedad de los archivos generados por ncu (root) al usuario actual
REAL_USER="${SUDO_USER:-$USER}"
chown -R "$REAL_USER" "$OUTDIR" 2>/dev/null || true
python3 plot_perfilado.py

echo ""
echo "================================================================"
echo " Resultados guardados en: $OUTDIR/"
echo "  perf_seq.txt         — contadores perf SEQ (texto crudo)"
echo "  perf_simd.txt        — contadores perf SIMD (texto crudo)"
echo "  ncu_cuda_full.txt    — métricas ncu legibles"
echo "  ncu_cuda_report.ncu-rep — reporte binario (Nsight Compute GUI)"
echo "  perf_metrics.csv     — métricas CPU tabuladas (SEQ vs SIMD)"
echo "  ncu_metrics.csv      — métricas GPU tabuladas (kernel CUDA)"
echo "  graficas/            — gráficas PNG generadas"
echo "================================================================"
