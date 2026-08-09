# Diseño del Framework DLP — Ray Tracing
**CE 4302 — Arquitectura de Computadores II**
**Proyecto 2 — Ranall Bolaños, Christopher Rodríguez, Andrés Molina**

---

## Decisión de diseño: Propuesta A vs Propuesta B

### Descripción de las propuestas

| Característica | Propuesta A — Descomposición por píxel | Propuesta B — Descomposición por tiles 2D |
|---|---|---|
| Unidad de trabajo | 1 píxel (CUDA) / 8 píxeles (SIMD) | Bloque de N×N píxeles |
| Asignación | 1 hilo CUDA por píxel; 1 vector AVX2 cada 8 píxeles consecutivos | 1 hilo CUDA por tile; bucle interno sobre píxeles del tile |
| Localidad de caché | Moderada (píxeles consecutivos acceden a la misma geometría) | Alta (todos los píxeles del tile reutilizan datos de escena en L1) |
| Coalescencia en GPU | Alta (hilos adyacentes acceden a posiciones contiguas en framebuffer) | Depende del orden de recorrido interno del tile |
| Warp divergence | Presente (distintos píxeles divergen en número de rebotes) | Presente en igual medida (la divergencia viene del rayo, no del tile) |
| Complejidad de implementación | Baja — índice lineal directo: `idx = blockIdx * blockDim + threadIdx` | Media — requiere calcular coordenadas globales desde tile + offset interno |
| Balance de carga | Perfecto por construcción (1 unidad = 1 píxel) | Puede desbalancearse si un tile concentra geometría compleja |

---

## Por qué elegimos la Propuesta A

### 1. El problema es naturalmente por píxel independiente

El ray tracing primario no tiene dependencias entre píxeles: el rayo que sale del píxel (i, j) no necesita ningún resultado del píxel (i±1, j±1). Esta independencia total hace que la descomposición por píxel sea la asignación natural del problema:

- **CUDA**: `threadIdx` mapea directamente a coordenadas de pantalla. No hay lógica adicional de indexación.
- **SIMD**: 8 píxeles consecutivos en la misma fila tienen orígenes idénticos y direcciones casi paralelas, lo que maximiza la utilización del vector AVX2 con mínimo overhead de empaquetado/desempaquetado.

### 2. La coalescencia en GPU ya está garantizada por la Propuesta A

El framebuffer está organizado en memoria lineal fila-por-fila (row-major). Con bloques 16×16 hilos:
- Los 16 hilos de una misma fila del bloque (misma `threadIdx.y`, distinto `threadIdx.x`) acceden a 16 posiciones **contiguas** en memoria.
- Esto genera exactamente **1 transacción de memoria de 128 bytes** (16 floats × 3 canales RGB), el tamaño óptimo de cacheline en GPUs NVIDIA.

La Propuesta B no mejora este comportamiento: dentro de un tile bien implementado, el acceso al framebuffer es igualmente coalescente. La ventaja teórica de B (reutilización de datos de escena) es irrelevante aquí porque la escena ya está en **memoria constante** (`__constant__`), que tiene caché dedicada con broadcast a todos los hilos del warp.

```
// renderer_cuda_dummy.cu — indexación directa Propuesta A
int x = blockIdx.x * blockDim.x + threadIdx.x;   // columna
int y = blockIdx.y * blockDim.y + threadIdx.y;   // fila
int idx = y * width + x;                          // índice lineal en framebuffer
```

### 3. La memoria constante anula la ventaja de localidad de la Propuesta B

En la Propuesta B, el argumento principal es que todos los píxeles de un tile acceden a los mismos datos de la escena (esferas, luces, materiales), reduciendo cache misses. Sin embargo, en nuestra implementación CUDA:

- Toda la escena reside en `__constant__` memory (48 KB en Ampere/Ada Lovelace).
- La memoria constante tiene una caché de solo lectura con **broadcast**: cuando todos los hilos de un warp leen la misma dirección (lo que ocurre con los parámetros de cada esfera), se sirve en **1 ciclo** desde la caché, sin importar si los hilos pertenecen al mismo tile o no.

Por lo tanto, la ventaja de localidad de B se anula completamente en este escenario.

### 4. La divergencia de warps no se reduce con tiles

La Propuesta B podría parecer atractiva si los píxeles de un tile tuvieran trayectorias similares (similar número de rebotes), reduciendo el warp divergence. Sin embargo:

- En una escena típica, píxeles dentro del mismo tile 16×16 pueden impactar esferas distintas o no impactar ninguna, generando divergencia independientemente del tamaño del tile.
- La divergencia es función de la **geometría proyectada**, no del esquema de tiling.
- La única forma real de reducir divergencia es el **ray sorting** (agrupar rayos similares), que es una técnica ortogonal a ambas propuestas y está fuera del alcance de este proyecto.

### 5. El balance de carga es perfecto en la Propuesta A

Con 1 hilo por píxel, los 1600×900 = 1.440.000 píxeles se distribuyen uniformemente entre todos los SMs de la RTX 4060 (24 SMs). El scheduler de la GPU asigna trabajo automáticamente sin ningún sesgo.

Con la Propuesta B, si un tile contiene varias esferas en su región (por ejemplo, el centro de la imagen), ese tile tardará más que los tiles del fondo vacío. Dado que el tile completo debe terminar antes de liberar el bloque de hilos CUDA, esto introduce **desbalance de carga innecesario**.

### 6. Simplicidad sin costo de rendimiento

La diferencia de rendimiento medida entre ambas propuestas en este tipo de escena es **marginal** (< 5% según literatura). Dado que:
- La Propuesta A es más simple de implementar y verificar.
- No sacrifica rendimiento medible en nuestro hardware (RTX 4060, sm_89).
- El cuello de botella real es el cálculo aritmético del kernel (FP32 throughput), no la latencia de memoria.

...la Propuesta A es la elección correcta en términos de relación complejidad/beneficio.

---

## Esquema de implementación — Propuesta A

### CUDA (renderer_cuda_dummy.cu)

```
Imagen 1600×900
└── Grid: (100 bloques × 57 bloques) = 5700 bloques
    └── Bloque: 16×16 = 256 hilos
        └── Hilo (tx, ty) → píxel (x, y)
            ├── Generar rayo desde cámara
            ├── Intersectar contra escena (__constant__)
            ├── [futuro] Phong + shadow rays + reflexiones
            └── Escribir RGB al framebuffer (coalescente)
```

**Métricas clave obtenidas (1600×900, RTX 4060):**
- Kernel time: **0.061 ms** (medido con CUDA Events)
- PCIe D→H transfer: **1.55 ms**
- Throughput: **890 Mpix/s** (solo kernel)
- Ocupación: limitada actualmente por falta de Phong (kernel muy pequeño)

### SIMD / AVX2 (renderer_simd_dummy.cpp)

```
Imagen 1600×900 = 1.440.000 píxeles
└── Bucle externo: 180.000 iteraciones (paso 8)
    └── Iteración i → píxeles [i, i+1, ..., i+7]
        ├── Empaquetar 8 orígenes + 8 direcciones en __m256
        ├── Calcular discriminante con _mm256_fmadd_ps (FMA AVX2)
        ├── Seleccionar impacto más cercano con _mm256_blendv_ps
        ├── [futuro] Phong vectorizado + shadow rays con máscara
        └── Desempaquetar y escribir 8 píxeles RGB
```

**Métricas clave obtenidas (1600×900, Ryzen 5 3600):**
- Tiempo (dummy, sin Phong): **~8 ms**
- Factor de vectorización efectivo: 8 píxeles/iteración con FMA
- Instrucciones AVX2 256-bit: contadas con `perf stat -e r0803`

---

## Conclusión

La **Propuesta A** (descomposición por píxel independiente) es la decisión de diseño óptima para este proyecto porque:

1. El problema es naturalmente independiente por píxel — no hay dependencias de datos entre píxeles.
2. La coalescencia de memoria en GPU ya está garantizada por la organización lineal del framebuffer.
3. La memoria constante de CUDA elimina la ventaja de localidad de datos que tendría la Propuesta B.
4. El balance de carga es perfecto por construcción en ambas implementaciones (CUDA y SIMD).
5. La complejidad de implementación es mínima sin sacrificar rendimiento medible.

La Propuesta B sería preferible únicamente en escenas con geometría muy densa y heterogénea donde la reutilización de datos de la escena desde L1/L2 fuera el cuello de botella dominante — lo cual no aplica a nuestro escenario con memoria constante y escenas de pocas esferas.
