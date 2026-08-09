# =============================================================================
# benchmark.ps1 — Métricas de rendimiento para el DLP Ray Tracing Framework
# CE 4302 Arquitectura de Computadores II
#
# Calcula:
#   a) Tiempo total de ejecución (SEQ, SIMD, CUDA) por resolución.
#   b) Speedup respecto a la versión secuencial.
#   c) Eficiencia paralela (SIMD: speedup/8 lanes | CUDA: speedup/N_SM).
#   d) Escalabilidad en función del tamaño del problema.
#
# Uso:
#   cd proyecto_2
#   .\benchmark.ps1
#
# Requisito: compilar primero con
#   mingw32-make all
#   mingw32-make cuda
# =============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Parámetros de hardware (RTX 4060 + Ryzen 5 3600)
# ---------------------------------------------------------------------------
$SIMD_LANES  = 8          # Ancho de registro AVX2 en floats
$CUDA_SMS    = 24          # Streaming Multiprocessors del RTX 4060

# ---------------------------------------------------------------------------
# Resoluciones para el análisis de escalabilidad
# Mantenemos relación de aspecto 16:9 y aumentamos el "tamaño del problema"
# ---------------------------------------------------------------------------
$RESOLUTIONS = @(
    @{ W = 400;  H = 225  },
    @{ W = 800;  H = 450  },
    @{ W = 1600; H = 900  },
    @{ W = 3200; H = 1800 }
)

# Directorio de binarios (relativo a proyecto_2/)
$BIN_SEQ  = ".\bin\renderer_seq.exe"
$BIN_SIMD = ".\bin\renderer_simd_dummy.exe"
$BIN_CUDA = ".\bin\renderer_cuda_dummy.exe"

# ---------------------------------------------------------------------------
# Directorios de resultados
# ---------------------------------------------------------------------------
$RESULTS_DIR  = "resultados"
$TABLAS_DIR   = "$RESULTS_DIR\tablas"
$GRAFICAS_DIR = "$RESULTS_DIR\graficas"
New-Item -ItemType Directory -Force -Path $TABLAS_DIR   | Out-Null
New-Item -ItemType Directory -Force -Path $GRAFICAS_DIR | Out-Null

# ---------------------------------------------------------------------------
# Función auxiliar: ejecuta un binario, captura su salida y extrae
# el tiempo en ms usando el patrón "time: X ms"
# Si el tiempo es 0 o muy pequeño, repite N veces y promedia
# ---------------------------------------------------------------------------
function Get-RenderTime {
    param(
        [string]$Bin,
        [int]$W,
        [int]$H,
        [string]$Output = "NUL",
        [int]$Repeats = 3
    )
    $times = @()
    for ($r = 0; $r -lt $Repeats; $r++) {
        $raw = & $Bin $W $H $Output 2>&1 | Out-String
        # CUDA imprime "TOTAL: X ms" — usar como tiempo total (kernel + PCIe)
        if ($raw -match 'TOTAL:\s*([\d.]+)\s*ms') {
            $times += [double]$Matches[1]
        } elseif ($raw -match 'time:\s*([\d.]+)\s*ms') {
            $times += [double]$Matches[1]
        } else {
            throw "No se pudo extraer el tiempo de: $raw"
        }
    }
    return ($times | Measure-Object -Average).Average
}

# ---------------------------------------------------------------------------
# Función para imprimir una línea de tabla con ancho fijo
# ---------------------------------------------------------------------------
function Write-Row {
    param([string[]]$Cols, [int[]]$Widths)
    $line = ""
    for ($i = 0; $i -lt $Cols.Count; $i++) {
        $line += $Cols[$i].PadLeft($Widths[$i]) + "  "
    }
    Write-Host $line
}

function Write-Separator {
    param([int[]]$Widths)
    $total = ($Widths | Measure-Object -Sum).Sum + $Widths.Count * 2
    Write-Host ("-" * $total)
}

# ---------------------------------------------------------------------------
# Verificar que los binarios existen
# ---------------------------------------------------------------------------
foreach ($b in @($BIN_SEQ, $BIN_SIMD, $BIN_CUDA)) {
    if (-not (Test-Path $b)) {
        Write-Error "Binario no encontrado: $b`nEjecuta: mingw32-make all; mingw32-make cuda"
    }
}

# ===========================================================================
# TABLA 1 — Escalabilidad: tiempo vs. tamaño del problema
# (Corre SEQ, SIMD y CUDA dummy a las 4 resoluciones)
# ===========================================================================
Write-Host ""
Write-Host "==========================================================="
Write-Host "  ESCALABILIDAD — Tiempo de ejecución vs. tamaño (Demo 2)"
Write-Host "  Nota: SIMD/CUDA solo hacen intersección+albedo (sin Phong)"
Write-Host "==========================================================="

$hdr  = @("Resolución", "Mpix",   "T_SEQ(ms)", "T_SIMD(ms)", "T_CUDA(ms)")
$wths = @(14,           8,        12,           13,            13)
Write-Host ""
Write-Row $hdr $wths
Write-Separator $wths

$scalability = @()   # guardar para tabla 2

foreach ($res in $RESOLUTIONS) {
    $W  = $res.W
    $H  = $res.H
    $mp = [math]::Round($W * $H / 1e6, 3)

    Write-Host "  Midiendo ${W}x${H}..." -NoNewline

    $t_seq  = Get-RenderTime -Bin $BIN_SEQ  -W $W -H $H -Output "NUL"
    $t_simd = Get-RenderTime -Bin $BIN_SIMD -W $W -H $H -Output "NUL"
    $t_cuda = Get-RenderTime -Bin $BIN_CUDA -W $W -H $H -Output "NUL"

    Write-Host " OK"

    $scalability += @{
        Label = "${W}x${H}"
        Mpix  = $mp
        Seq   = $t_seq
        Simd  = $t_simd
        Cuda  = $t_cuda
    }

    Write-Row @(
        "${W}x${H}",
        "$mp",
        "$([math]::Round($t_seq,  2))",
        "$([math]::Round($t_simd, 3))",
        "$([math]::Round($t_cuda, 3))"
    ) $wths
}

# ===========================================================================
# TABLA 2 — Speedup y Eficiencia a 1600×900
# ===========================================================================
$ref = $scalability | Where-Object { $_.Label -eq "1600x900" }
$t_seq_ref  = $ref.Seq
$t_simd_ref = $ref.Simd
$t_cuda_ref = $ref.Cuda

$speedup_simd = $t_seq_ref / $t_simd_ref
$speedup_cuda = $t_seq_ref / $t_cuda_ref

# Eficiencia paralela:
#   SIMD: E = Speedup / 8       (ideal = 8× con 8 lanes AVX2)       → %
#   CUDA: E = Speedup / N_SM   (ideal = N_SM× con 24 SMs)           → %
#
# Nota: como SIMD/CUDA hacen menos trabajo que SEQ (sin Phong),
# el speedup medido es mayor al esperado. Se documenta como
# "intersección-only" y será recalculado con el algoritmo completo.
$eff_simd = [math]::Min($speedup_simd / $SIMD_LANES * 100, 9999)
$eff_cuda = [math]::Min($speedup_cuda / $CUDA_SMS   * 100, 9999)

Write-Host ""
Write-Host "==========================================================="
Write-Host "  SPEEDUP y EFICIENCIA PARALELA a 1600x900"
Write-Host "  (comparación intersección-only: SIMD y CUDA vs. SEQ full)"
Write-Host "  * Para comparación justa: implementar Phong en SIMD/CUDA"
Write-Host "==========================================================="
Write-Host ""

$hdr2  = @("Impl.",  "T(ms)",                     "Speedup vs SEQ",                     "Eficiencia",                          "Referencia paralela")
$wths2 = @(8,        10,                           17,                                   13,                                    22)
Write-Row $hdr2 $wths2
Write-Separator $wths2

Write-Row @(
    "SEQ",
    "$([math]::Round($t_seq_ref, 2))",
    "1.00×  (baseline)",
    "100.0%",
    "1 hilo"
) $wths2

Write-Row @(
    "SIMD",
    "$([math]::Round($t_simd_ref, 3))",
    "$([math]::Round($speedup_simd, 2))×",
    "$([math]::Round($eff_simd, 1))%",
    "8 lanes AVX2"
) $wths2

Write-Row @(
    "CUDA",
    "$([math]::Round($t_cuda_ref, 3))",
    "$([math]::Round($speedup_cuda, 2))×",
    "$([math]::Round($eff_cuda, 1))%",
    "$CUDA_SMS SMs RTX 4060"
) $wths2

# ===========================================================================
# TABLA 3 — Escalabilidad normalizada (T / T_baseline @ 400x225)
# ===========================================================================
Write-Host ""
Write-Host "==========================================================="
Write-Host "  ESCALABILIDAD NORMALIZADA (T relativo a 400x225)"
Write-Host "  Ideal: duplicar resolución → 4× el tiempo (O(W·H))"
Write-Host "==========================================================="
Write-Host ""

$hdr3  = @("Resolución", "Factor pix", "T_SEQ rel.", "T_SIMD rel.", "T_CUDA rel.")
$wths3 = @(14,           12,           12,            13,             13)
Write-Row $hdr3 $wths3
Write-Separator $wths3

$base_seq  = ($scalability | Where-Object { $_.Label -eq "400x225" }).Seq
$base_simd = ($scalability | Where-Object { $_.Label -eq "400x225" }).Simd
$base_cuda = ($scalability | Where-Object { $_.Label -eq "400x225" }).Cuda

foreach ($s in $scalability) {
    $factor = [math]::Round($s.Mpix / ($scalability[0].Mpix), 2)
    $rel_seq  = if ($base_seq  -gt 0) { "$([math]::Round($s.Seq  / $base_seq,  2))x" } else { "-" }
    $rel_simd = if ($base_simd -gt 0.001) { "$([math]::Round($s.Simd / $base_simd, 2))x" } else { "<0.1ms" }
    $rel_cuda = if ($base_cuda -gt 0) { "$([math]::Round($s.Cuda / $base_cuda, 2))x" } else { "-" }
    Write-Row @(
        $s.Label,
        "${factor}x",
        $rel_seq,
        $rel_simd,
        $rel_cuda
    ) $wths3
}

# ===========================================================================
# EXPORTAR CSVs a resultados/tablas/
# ===========================================================================

# CSV 1: tiempo de ejecucion por resolución
$csv1 = "Resolucion,Mpix,T_SEQ_ms,T_SIMD_ms,T_CUDA_ms"
foreach ($s in $scalability) {
    $csv1 += "`n$($s.Label),$($s.Mpix),$([math]::Round($s.Seq,3)),$([math]::Round($s.Simd,4)),$([math]::Round($s.Cuda,4))"
}
Set-Content -Path "$TABLAS_DIR\escalabilidad.csv" -Value $csv1 -Encoding UTF8

# CSV 2: speedup y eficiencia a 1600x900
$csv2 = "Impl,T_ms,Speedup,Eficiencia_pct,Referencia"
$csv2 += "`nSEQ,$([math]::Round($t_seq_ref,3)),1.0,100.0,1 hilo"
$csv2 += "`nSIMD,$([math]::Round($t_simd_ref,4)),$([math]::Round($speedup_simd,4)),$([math]::Round($eff_simd,2)),8 lanes AVX2"
$csv2 += "`nCUDA,$([math]::Round($t_cuda_ref,4)),$([math]::Round($speedup_cuda,4)),$([math]::Round($eff_cuda,2)),$CUDA_SMS SMs RTX 4060"
Set-Content -Path "$TABLAS_DIR\speedup_eficiencia.csv" -Value $csv2 -Encoding UTF8

# CSV 3: escalabilidad normalizada
$csv3 = "Resolucion,Mpix,Factor_pix,T_SEQ_rel,T_SIMD_rel,T_CUDA_rel"
foreach ($s in $scalability) {
    $factor = [math]::Round($s.Mpix / $scalability[0].Mpix, 2)
    $r_seq  = if ($base_seq  -gt 0)     { [math]::Round($s.Seq  / $base_seq,  4) } else { 0 }
    $r_simd = if ($base_simd -gt 0.001) { [math]::Round($s.Simd / $base_simd, 4) } else { 0 }
    $r_cuda = if ($base_cuda -gt 0)     { [math]::Round($s.Cuda / $base_cuda, 4) } else { 0 }
    $csv3 += "`n$($s.Label),$($s.Mpix),$factor,$r_seq,$r_simd,$r_cuda"
}
Set-Content -Path "$TABLAS_DIR\escalabilidad_normalizada.csv" -Value $csv3 -Encoding UTF8

Write-Host ""
Write-Host "Tablas CSV exportadas -> $TABLAS_DIR\"

# ===========================================================================
# DISTRIBUCIÓN — $N_DIST corridas por (impl × resolución) para IC 95% y boxplot
# ===========================================================================
$N_DIST = 20
Write-Host ""
Write-Host "==========================================================="
Write-Host "  DISTRIBUCIÓN DE TIEMPOS ($N_DIST corridas por configuración)"
Write-Host "  Calcula: media, desviación estándar, IC 95% (z = 1.96)"
Write-Host "==========================================================="

$dist_rows = @()
$stat_rows = @()

foreach ($res in $RESOLUTIONS) {
    $W = $res.W; $H = $res.H
    Write-Host "  ${W}x${H}..." -NoNewline

    foreach ($impl_name in @("SEQ", "SIMD", "CUDA")) {
        $bin = switch ($impl_name) {
            "SEQ"  { $BIN_SEQ  }
            "SIMD" { $BIN_SIMD }
            "CUDA" { $BIN_CUDA }
        }

        $times = @()
        for ($i = 0; $i -lt $N_DIST; $i++) {
            $raw = & $bin $W $H NUL 2>&1 | Out-String
            $t = 0.0
            if ($raw -match 'TOTAL:\s*([\d.]+)\s*ms') {
                $t = [double]$Matches[1]
            } elseif ($raw -match 'time:\s*([\d.]+)\s*ms') {
                $t = [double]$Matches[1]
            }
            if ($t -gt 0) { $times += $t }
            $dist_rows += "$impl_name,${W}x${H},$i,$([math]::Round($t, 4))"
        }

        # Estadísticas descriptivas (muestra, ddof=1)
        $n_t    = $times.Count
        $mean_t = if ($n_t -gt 0) { ($times | Measure-Object -Average).Average } else { 0.0 }
        $std_t  = 0.0
        if ($n_t -gt 1) {
            $sq_arr = $times | ForEach-Object { [math]::Pow($_ - $mean_t, 2) }
            $std_t  = [math]::Sqrt(($sq_arr | Measure-Object -Sum).Sum / ($n_t - 1))
        }
        $sem_t   = if ($n_t -gt 0) { $std_t / [math]::Sqrt($n_t) } else { 0.0 }
        $ci_half = 1.96 * $sem_t
        $ci_lo   = [math]::Round($mean_t - $ci_half, 4)
        $ci_hi   = [math]::Round($mean_t + $ci_half, 4)
        $stat_rows += "$impl_name,${W}x${H},$n_t,$([math]::Round($mean_t,4)),$([math]::Round($std_t,4)),$ci_lo,$ci_hi"
    }
    Write-Host " OK"
}

# Imprimir tabla de estadísticas
Write-Host ""
$hdr_s  = @("Impl", "Resolución",  "n",  "Media(ms)",  "StdDev",  "CI95_low",  "CI95_high")
$wths_s = @(6,      12,            4,    12,            10,        10,           10)
Write-Row $hdr_s $wths_s
Write-Separator $wths_s
foreach ($sr in $stat_rows) {
    $c = $sr.Split(",")
    Write-Row @($c[0], $c[1], $c[2], $c[3], $c[4], $c[5], $c[6]) $wths_s
}

# Exportar CSVs
$csv4 = "Impl,Resolucion,Run,T_ms`n" + ($dist_rows -join "`n")
Set-Content -Path "$TABLAS_DIR\distribuciones.csv" -Value $csv4 -Encoding UTF8

$csv5 = "Impl,Resolucion,N,Media_ms,StdDev_ms,CI95_low,CI95_high`n" + ($stat_rows -join "`n")
Set-Content -Path "$TABLAS_DIR\estadisticas.csv" -Value $csv5 -Encoding UTF8
Write-Host ""
Write-Host "Estadísticas exportadas -> $TABLAS_DIR\"

Write-Host ""
Write-Host "Benchmark completado."
