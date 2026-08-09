# DLP Ray Tracing Framework

**CE 4302 — Arquitectura de Computadores II**  
Grupo: Randall Bolaños · Christopher Rodríguez · Andrés Molina

---

## Requisitos

```bash
# Compilador C++17 con soporte AVX2
sudo apt install build-essential

# CUDA Toolkit (para renderer CUDA)
sudo apt install nvidia-cuda-toolkit --fix-missing

# Python con matplotlib (para gráficas)
pip install matplotlib numpy
```

---

## Flujo completo en un solo paso

```bash
cd proyecto_2

# 1. Compilar los 3 renderers
make all

# 2. Ejecutar los 3 y generar imágenes PPM
make run_all

# 3. Benchmark + gráficas
chmod +x benchmark.sh
./benchmark.sh
```

Esto genera:
- `output_seq.ppm`, `output_simd_dummy.ppm`, `output_cuda_dummy.ppm` — imágenes renderizadas
- `resultados/tablas/*.csv` — datos de rendimiento
- `resultados/graficas/*.png` — gráficas

---

## Comandos individuales

### Compilar

```bash
make seq        # Solo secuencial
make simd       # Solo SIMD/AVX2
make cuda       # Solo CUDA
make all        # Los 3
```

### Ejecutar

```bash
make run_seq    # SEQ a 1600x900
make run_simd   # SIMD a 1600x900
make run_cuda   # CUDA a 1600x900
make run_all    # Los 3 en secuencia
```

Con resolución personalizada:

```bash
./bin/renderer_seq          400 225 salida.ppm
./bin/renderer_simd_dummy   800 450 salida.ppm
./bin/renderer_cuda_dummy  1600 900 salida.ppm
```

### Benchmark completo (mide, exporta CSVs y genera gráficas)

```bash
./benchmark.sh
```

### Solo regenerar gráficas (sin volver a medir)

```bash
python3 plot_results.py
```

### Perfilado de desempeño (perf + ncu + nsys)

```bash
chmod +x perfil.sh
sudo ./perfil.sh
```

Resultados en `resultados/perfilado/`.

---

## Ver imágenes generadas

```bash
eog output_seq.ppm &
eog output_simd_dummy.ppm &
eog output_cuda_dummy.ppm &
```

O con cualquier visor compatible con PPM (`feh`, `display`, `gimp`).

---

## Estructura del proyecto

```
proyecto_2/
├── include/                      # Headers compartidos (vec3, ray, hit, material, sphere, light, camera, scene)
├── src/
│   ├── sequential/renderer_seq.cpp         # SEQ: Phong + sombras + reflexión
│   ├── simd/renderer_simd_dummy.cpp        # SIMD: intersección AVX2 (8 rayos/ciclo)
│   └── cuda/renderer_cuda_dummy.cu         # CUDA: 1 hilo/píxel, bloques 16×16
├── bin/                          # Binarios compilados (generado por make)
├── resultados/
│   ├── tablas/                   # CSVs: estadisticas, speedup, escalabilidad
│   └── graficas/                 # PNGs generados por plot_results.py
│   └── perfilado/                # Salidas de perf / ncu / nsys
├── Makefile
├── benchmark.sh                  # Benchmark Linux (equivalente a benchmark.ps1)
├── benchmark.ps1                 # Benchmark Windows (PowerShell)
├── plot_results.py               # Genera gráficas desde los CSVs
├── perfil.sh                     # Perfilado con perf, ncu y nsys
└── Progreso.md                   # Estado de implementación y documentación técnica
```

---

## Resultados Demo 2 (prototipos)

| Impl. | Algoritmo | Tiempo 1600×900 | Speedup vs SEQ |
|-------|-----------|-----------------|----------------|
| SEQ   | Phong + sombras + reflexión (depth=3) | ~151 ms | 1× |
| SIMD  | Solo intersección AVX2 | ~8.1 ms | 18.6× |
| CUDA  | Solo intersección GPU  | ~2.4 ms | 63× |

> **Nota:** SIMD y CUDA son prototipos dummy. El speedup no es comparable directamente con SEQ porque no implementan iluminación Phong ni sombras. El speedup real se medirá en la implementación completa.

---

## Hardware objetivo

- **CPU:** AMD Ryzen 7 7700 (Zen 4, 8 núcleos, AVX2/FMA)
- **GPU:** NVIDIA GeForce RTX 4060 (SM 8.9, 24 SMs, 8 GB VRAM)
- **OS:** Ubuntu 24.04 LTS
