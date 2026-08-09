# Progreso del Proyecto — DLP Ray Tracing Framework

**CE 4302 Arquitectura de Computadores II**  
Grupo: Randall Bolaños · Christopher Rodríguez · Andrés Molina  
Fecha de actualización: 24 mayo 2026

---

## Cómo ejecutar cada implementación

> **Requisito previo:** Abrir una terminal en la carpeta `proyecto_2/`.
> En Windows usar **`mingw32-make`** en lugar de `make`.

### Secuencial (baseline)

```powershell
mingw32-make run_seq
# Genera: output_seq.ppm
```

### SIMD / AVX2 (prototipo dummy)

```powershell
mingw32-make run_simd
# Genera: output_simd_dummy.ppm
```

### CUDA (prototipo dummy)

```powershell
mingw32-make run_cuda
# Genera: output_cuda_dummy.ppm
```

> `nvcc` no está en el PATH por defecto — el Makefile ya apunta a la ruta completa de
> `CUDA v13.2` y usa `cl.exe` de Visual Studio 2022 como host compiler. No se requiere
> configuración adicional.

Para compilar sin ejecutar:

```powershell
mingw32-make seq
mingw32-make simd
mingw32-make cuda
```

Para limpiar todos los binarios:

```powershell
mingw32-make clean
```

---

## Estructura del proyecto

```
proyecto_2/
├── include/                  # Cabeceras compartidas entre todas las implementaciones
│   ├── vec3.h                # Vector 3D flotante (dot, cross, normalize, reflect)
│   ├── ray.h                 # Rayo paramétrico P(t) = origin + t·direction
│   ├── hit.h                 # Registro de intersección (punto, normal, material, t)
│   ├── material.h            # Propiedades de superficie Phong (Ka, Kd, Ks, shininess)
│   ├── sphere.h              # Primitiva esfera con intersección rayo-esfera
│   ├── light.h               # Fuente de luz puntual (posición, intensidad)
│   ├── camera.h              # Cámara perspectiva look-at
│   └── scene.h               # Contenedor de escena (esferas + luces)
├── src/
│   ├── sequential/
│   │   └── renderer_seq.cpp  # ✅ Implementación secuencial completa
│   ├── simd/
│   │   └── renderer_simd_dummy.cpp  # ✅ Prototipo AVX2 (intersección)
│   └── cuda/
│       └── renderer_cuda_dummy.cu   # ✅ Prototipo kernel CUDA
├── bin/                      # Binarios compilados (generado por make)
├── Makefile
└── Progreso.md               # Este archivo
```

---

## Implementaciones actuales

### ✅ SEQ — Renderer Secuencial (`renderer_seq.cpp`)

**Estado:** Completo y verificado · **Tiempo:** ~205 ms en 1600×900

Renderer de referencia que implementa el algoritmo completo de ray tracing con modelo de iluminación Phong.

**Algoritmo:**
1. Para cada píxel (1600×900), generar un rayo primario desde la cámara.
2. Buscar la intersección más cercana con las 5 esferas de la escena.
3. Si hay intersección, calcular el color con shading Phong:
   - **Ambiental:** `Ka · albedo · luz_ambiental`
   - **Difusa:** `Kd · max(0, N·L) · intensidad_luz · albedo`
   - **Especular:** `Ks · (R·V)^shininess · intensidad_luz`
4. Para cada fuente de luz, lanzar un **rayo de sombra** para verificar oclusión.
5. Si el material tiene `reflectivity > 0`, trazar rayos recursivos (profundidad máx. = 3).
6. Aplicar corrección gamma 2.0 (raíz cuadrada) y escribir PPM binario.

**Escena (`scene.h`):**
| Esfera | Centro | Radio | Albedo |
|--------|--------|-------|--------|
| Suelo  | (0, −100.5, −2) | 100 | Gris |
| Centro | (0, 0, −2) | 0.5 | Rojo |
| Izquierda | (−1, 0, −2) | 0.5 | Azul |
| Derecha | (1, 0, −2) | 0.5 | Amarillo |
| Arriba (espejo) | (0, 1, −2) | 0.35 | Blanco, reflectividad 0.9 |

**Fuentes de luz:** Blanca principal (0, 3, 1) + Azul de relleno (−2, 2, 0)

**Cámara:** posición (0, 0.5, 3), look-at (0, 0, −2), vfov = 50°

---

### ✅ SIMD — Prototipo AVX2 (`renderer_simd_dummy.cpp`)

**Estado:** Prototipo funcional (Demo 2) · **Tiempo:** ~7 ms en 1600×900

Demuestra el esquema DLP en CPU: procesar **8 píxeles en paralelo** usando registros `__m256` de AVX2.

**Estructuras clave:**

```cpp
struct Ray8 {
    __m256 ox, oy, oz;   // orígenes (idénticos — todos desde la cámara)
    __m256 dx, dy, dz;   // 8 direcciones distintas (un píxel diferente cada una)
};
```

**Función `intersect_8_sphere()`:**
- Calcula el discriminante para 8 rayos simultáneamente usando instrucciones FMA:
  ```
  disc = hb·hb − a·c   (con _mm256_fmadd_ps)
  ```
- `_mm256_cmp_ps` genera una máscara de hits (1 = intersección válida, 0 = miss).
- `_mm256_blendv_ps` selecciona el color del albedo vs. fondo según la máscara.

**Bucle principal:**
```
for cada fila y:
    for cada columna x en pasos de 8:
        generar Ray8 (8 rayos escalares empaquetados en __m256)
        para cada esfera:
            intersect_8_sphere() → máscara + t
            actualizar t_min con blendv (tomar el hit más cercano)
        escribir 8 píxeles
```

**Limitaciones del prototipo:** Sin iluminación Phong, sin rayos de sombra, sin reflexiones.  
El color por píxel = albedo de la esfera más cercana, o color de fondo.

---

### ✅ CUDA — Prototipo kernel (`renderer_cuda_dummy.cu`)

**Estado:** Prototipo funcional (Demo 2) · **Tiempo:** ~0.06 ms (kernel puro, sin PCIe)  
**GPU:** NVIDIA GeForce RTX 4060 · SM 8.9 · 24 SMs · 8 GB VRAM

Demuestra el esquema DLP en GPU: **1 hilo CUDA por píxel**, lanzados en bloques 2D de 16×16.

**Configuración de lanzamiento:**
```
Block: 16×16 = 256 hilos
Grid:  100×57 = 5700 bloques
Total: 1.459.200 hilos (> 1600×900 = 1.440.000 píxeles, con guarda de límites)
```

**Diseño de memoria:**

| Recurso | Tipo | Justificación |
|---------|------|---------------|
| Escena (esferas, cámara) | `__constant__` | Solo lectura broadcast — todos los hilos del warp leen el mismo dato → sin conflictos de banco |
| Framebuffer (R, G, B) | SoA — 3 arrays separados | Hilos adyacentes (warp) acceden a posiciones contiguas → accesos coalescentes → máximo ancho de banda |

**Flujo del kernel `ray_trace_kernel`:**
1. Calcular `(px, py)` desde `blockIdx` y `threadIdx`.
2. Calcular coordenadas normalizadas `u`, `v` con flip de Y.
3. Generar dirección del rayo desde los parámetros de cámara en `__constant__`.
4. Bucle sobre esferas: intersección rayo-esfera (misma matemática que SEQ).
5. Escribir color al framebuffer SoA: `fb_r[idx]`, `fb_g[idx]`, `fb_b[idx]`.

**Medición de tiempo:** CUDA Events (excluye transferencias PCIe).

**Limitaciones del prototipo:** Sin iluminación Phong, sin rayos de sombra, sin reflexiones.

---

## Resumen de rendimiento (Demo 2)

| Impl. | Algoritmo | Tiempo 1600×900 | Factor vs SEQ |
|-------|-----------|-----------------|---------------|
| SEQ | Phong + sombras + reflexión (depth=3) | ~205 ms | 1× (baseline) |
| SIMD dummy | Solo intersección + albedo | ~7 ms | ~29× |
| CUDA dummy | Solo intersección + albedo (kernel) | ~0.06 ms | ~3400× |

> ⚠️ SIMD y CUDA no son comparables directamente con SEQ todavía, ya que no implementan
> el shading completo. El speedup real se medirá cuando las tres implementaciones ejecuten
> el mismo algoritmo (Phong + sombras + reflexiones).

---

## Notas técnicas — Compilación en Windows

- **Compilador C++:** g++ (MinGW-W64 x86_64) 13.2.0
- **CUDA Toolkit:** v13.2 · nvcc en `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/`
- **Host compiler para nvcc:** MSVC `cl.exe` (Visual Studio 2022 Community, 14.42.34433)
  - MinGW g++ no es compatible como host de nvcc en Windows.
- **Build tool:** `mingw32-make` 4.4.1 (no `make`)
- **Detección de SO:** El Makefile ajusta automáticamente `mkdir` y `rm` entre Windows y Linux/WSL.
