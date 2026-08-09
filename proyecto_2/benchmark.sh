#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — Benchmark de rendimiento para el DLP Ray Tracing Framework
# CE 4302 Arquitectura de Computadores II
#
# Equivalente bash de benchmark.ps1 para Linux.
# Calcula:
#   - Tiempo promedio (N repeticiones) por implementación y resolución
#   - Speedup y eficiencia paralela
# Exporta CSVs a resultados/tablas/ listos para plot_results.py
#
# Uso:
#   chmod +x benchmark.sh
#   ./benchmark.sh
# =============================================================================

set -e

SIMD_LANES=8
CUDA_SMS=24
REPEATS=20

BIN_SEQ="./bin/renderer_seq"
BIN_SIMD="./bin/renderer_simd_dummy"
BIN_CUDA="./bin/renderer_cuda_dummy"

TABLAS="resultados/tablas"
mkdir -p "$TABLAS" "resultados/graficas"

RESOLUTIONS=("400 225" "800 450" "1600 900" "3200 1800")

# ---------------------------------------------------------------------------
# get_time <bin> <W> <H> <repeats>
# Devuelve el promedio de <repeats> mediciones (extrae "time: X ms" o "TOTAL: X ms")
# ---------------------------------------------------------------------------
get_time() {
    local BIN=$1 W=$2 H=$3 N=$4
    local sum=0
    for ((i=0; i<N; i++)); do
        local out
        out=$("$BIN" "$W" "$H" /dev/null 2>&1)
        local t
        # CUDA imprime "TOTAL: X ms"; SEQ/SIMD imprimen "time: X ms"
        if echo "$out" | grep -q "TOTAL:"; then
            t=$(echo "$out" | grep -oP 'TOTAL:\s*\K[\d.]+')
        else
            t=$(echo "$out" | grep -oP 'time:\s*\K[\d.]+')
        fi
        sum=$(echo "$sum + $t" | bc -l)
    done
    echo "scale=4; $sum / $N" | bc -l
}

# ---------------------------------------------------------------------------
# CSV 1 — estadisticas.csv (media, stddev, IC95)
# ---------------------------------------------------------------------------
STATS_CSV="$TABLAS/estadisticas.csv"
echo "Impl,Resolucion,N,Media_ms,StdDev_ms,CI95_low,CI95_high" > "$STATS_CSV"

# CSV 2 — escalabilidad.csv
ESCAL_CSV="$TABLAS/escalabilidad.csv"
echo "Resolucion,Mpix,T_SEQ_ms,T_SIMD_ms,T_CUDA_ms" > "$ESCAL_CSV"

# CSV 3 — distribuciones.csv (todas las muestras individuales)
DIST_CSV="$TABLAS/distribuciones.csv"
echo "Impl,Resolucion,Muestra,T_ms" > "$DIST_CSV"

echo "============================================================="
echo "  BENCHMARK — DLP Ray Tracing Framework (Linux)"
echo "  $(date)"
echo "============================================================="

for RES in "${RESOLUTIONS[@]}"; do
    W=$(echo "$RES" | cut -d' ' -f1)
    H=$(echo "$RES" | cut -d' ' -f2)
    LABEL="${W}x${H}"
    MPIX=$(echo "scale=3; $W * $H / 1000000" | bc -l)

    echo ""
    echo "── Resolución: $LABEL ──────────────────────────────────"

    for IMPL in SEQ SIMD CUDA; do
        case $IMPL in
            SEQ)  BIN="$BIN_SEQ"  ;;
            SIMD) BIN="$BIN_SIMD" ;;
            CUDA) BIN="$BIN_CUDA" ;;
        esac

        echo -n "  [$IMPL] Midiendo ($REPEATS muestras)..."

        # Recolectar muestras individuales
        SAMPLES=()
        for ((i=0; i<REPEATS; i++)); do
            OUT=$("$BIN" "$W" "$H" /dev/null 2>&1)
            if echo "$OUT" | grep -q "TOTAL:"; then
                T=$(echo "$OUT" | grep -oP 'TOTAL:\s*\K[\d.]+')
            else
                T=$(echo "$OUT" | grep -oP 'time:\s*\K[\d.]+')
            fi
            SAMPLES+=("$T")
            echo "$IMPL,$LABEL,$((i+1)),$T" >> "$DIST_CSV"
        done

        # Calcular media, stddev, IC95 con python (más limpio que bc para estadísticas)
        PY_SAMPLES=$(IFS=,; echo "${SAMPLES[*]}")
        STATS=$(python3 - <<EOF
import math, sys
samples = [${PY_SAMPLES}]
n = len(samples)
mean = sum(samples) / n
var  = sum((x - mean)**2 for x in samples) / (n - 1)
std  = math.sqrt(var)
# t-student 95% para n-1 grados de libertad (aproximación normal para n>=20)
t95  = 1.96 if n >= 30 else {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,
       10:2.228,15:2.131,19:2.093}.get(n-1, 2.093)
ci   = t95 * std / math.sqrt(n)
print(f"{mean:.4f},{std:.4f},{mean-ci:.4f},{mean+ci:.4f}")
EOF
)
        echo "$IMPL,$LABEL,$REPEATS,$STATS" >> "$STATS_CSV"
        MEAN=$(echo "$STATS" | cut -d',' -f1)
        echo " media=${MEAN} ms"

        # Guardar media por implementación para tabla de escalabilidad
        case $IMPL in
            SEQ)  T_SEQ=$MEAN  ;;
            SIMD) T_SIMD=$MEAN ;;
            CUDA) T_CUDA=$MEAN ;;
        esac
    done

    echo "$LABEL,$MPIX,$T_SEQ,$T_SIMD,$T_CUDA" >> "$ESCAL_CSV"
done

# ---------------------------------------------------------------------------
# CSV 4 — speedup_eficiencia.csv (a 1600x900)
# ---------------------------------------------------------------------------
REF_LINE=$(grep "^SEQ,1600x900" "$STATS_CSV")
T_SEQ_REF=$(echo "$REF_LINE" | cut -d',' -f4)
T_SIMD_REF=$(grep "^SIMD,1600x900" "$STATS_CSV" | cut -d',' -f4)
T_CUDA_REF=$(grep "^CUDA,1600x900" "$STATS_CSV" | cut -d',' -f4)

python3 - <<EOF > "$TABLAS/speedup_eficiencia.csv"
t_seq  = $T_SEQ_REF
t_simd = $T_SIMD_REF
t_cuda = $T_CUDA_REF
sp_simd = t_seq / t_simd
sp_cuda = t_seq / t_cuda
ef_simd = sp_simd / $SIMD_LANES * 100
ef_cuda = sp_cuda / $CUDA_SMS   * 100
print("Impl,T_ms,Speedup,Eficiencia_pct,Referencia")
print(f"SEQ,{t_seq:.3f},1.0,100.0,1 hilo")
print(f"SIMD,{t_simd:.4f},{sp_simd:.4f},{ef_simd:.2f},8 lanes AVX2")
print(f"CUDA,{t_cuda:.4f},{sp_cuda:.4f},{ef_cuda:.2f},{$CUDA_SMS} SMs RTX 4060")
EOF

# ---------------------------------------------------------------------------
# CSV 5 — escalabilidad_normalizada.csv (normalizada a 400x225)
# ---------------------------------------------------------------------------
python3 - <<EOF > "$TABLAS/escalabilidad_normalizada.csv"
import csv
rows = list(csv.DictReader(open("$ESCAL_CSV")))
base_seq  = float(rows[0]['T_SEQ_ms'])
base_simd = float(rows[0]['T_SIMD_ms'])
base_cuda = float(rows[0]['T_CUDA_ms'])
print("Resolucion,Factor_pix,T_SEQ_rel,T_SIMD_rel,T_CUDA_rel")
for r in rows:
    w, h = r['Resolucion'].split('x')
    mpix = float(w) * float(h) / (400*225)
    print(f"{r['Resolucion']},"
          f"{mpix:.4f},"
          f"{float(r['T_SEQ_ms'])/base_seq:.4f},"
          f"{float(r['T_SIMD_ms'])/base_simd:.4f},"
          f"{float(r['T_CUDA_ms'])/base_cuda:.4f}")
EOF

echo ""
echo "============================================================="
echo " CSVs generados en $TABLAS/"
echo "   estadisticas.csv, escalabilidad.csv, distribuciones.csv"
echo "   speedup_eficiencia.csv, escalabilidad_normalizada.csv"
echo "============================================================="
echo " Generando gráficas..."
python3 plot_results.py
echo " Gráficas guardadas en resultados/graficas/"
echo "============================================================="
