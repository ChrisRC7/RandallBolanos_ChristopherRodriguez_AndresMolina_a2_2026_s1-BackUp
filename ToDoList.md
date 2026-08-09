# ToDoList — Proyecto Grupal 2: Framework DLP con Ray Tracing
**CE 4302 — Arquitectura de Computadores II** | Entrega final: **4 de junio 2026**
Resolución objetivo: **1600×900** | Hardware: AMD Ryzen 5 3600 + NVIDIA GPU

---

## ⚠️ CRÍTICO — Observaciones del Profesor

- [x] Reemplazar `raytracing.github.io` en el paper por referencias académicas verificadas:
  - Whitted, T. (1980). "An improved illumination model for shaded display." *Commun. ACM*, 23(6), 343–349.
  - Pharr, M., Jakob, W., & Humphreys, G. (2023). *Physically Based Rendering*, 4th ed. MIT Press.
- [ ] Documentar en el paper el alcance delimitado: solo esferas, Phong/Lambertian, profundidad máx. 3 rebotes, resoluciones 256×256 → 1600×900

---

## 🔴 URGENTE — Demo 2: 26 mayo

- [x] Commits incrementales en el repo (no todo en un solo commit)
- [x] Actualizar diagramas del paper respecto a Demo 1 (indicar cambios significativos)
- [x] Identificar y documentar variables clave de paralelización (resolución, profundidad, tamaño de tile)
- [x] **[SEQ]** Implementación secuencial compilando y produciendo imagen correcta en 1600×900 ✅ (205 ms)
- [x] **[SIMD]** Prototipo dummy AVX2: intersección de 8 rayos en paralelo con `_mm256_fmadd_ps`, `_mm256_blendv_ps` ✅ (7 ms)
- [x] **[CUDA]** Prototipo dummy kernel: lanzar hilos y escribir framebuffer ✅ (0.06 ms kernel — RTX 4060, 24 SMs, sm_89)

---

## 🟡 IMPLEMENTACIÓN COMPLETA — Antes del 4 junio

### Sequential (Baseline)
- [x] Medir tiempo de ejecución con `chrono` para 4 resoluciones (256², 512², 1024², 1600×900)
- [x] Generar imagen de referencia PPM para validación de versiones paralelas

### SIMD / AVX2
- [x] Pipeline completo (8 rayos/ciclo): load → dirección → intersección → máscara → shading → store
- [x] Manejar residuo cuando WIDTH×HEIGHT no es múltiplo de 8 (scalar fallback)
- [x] Shadow rays con masking AVX2
- [x] Validar contra secuencial: diff pixel a pixel, tolerancia ≤ 1e-4
- [ ] Corregir bug restante de precisión/recurrencia detectado en la validación SIMD vs SEQ

### CUDA
- [x] Kernel completo: intersección rayo-esfera, Phong, shadow rays, 1 hilo/píxel, bloques 16×16
- [x] Escena en memoria constante (`__constant__`) para reducir latencia de acceso
- [x] Medir tiempo del kernel con CUDA Events (separar de transferencias PCIe)
- [x] Validar contra secuencial: diferencia numérica documentada y justificada (97/65536 px difieren, diff máx=190, causa: divergencia float32 en trace_ray recursivo CPU vs GPU — ver bitacora_gpu.md)

---

## 🟢 MÉTRICAS Y PERFILADO

- [x] Tabla de resultados: tiempo, speedup S=T_seq/T_k, eficiencia E=S/p — para las 3 versiones × 4 resoluciones
- [x] `perf stat` — contar instrucciones AVX2 emitidas (`fp_arith_inst_retired.256b_packed_single`), cache-misses L1/L2
- [x] Nsight Compute — ocupación de SMs, ancho de banda memoria global, instrucciones por warp
- [x] Nsight Systems — timeline: identificar cuello de botella kernel vs. transferencia PCIe
- [ ] Justificar en el paper cada métrica elegida y conclusión extraída del dato

---

## 📐 DISEÑO

- [ ] **Propuesta A**: descomposición por píxel independiente (1 hilo/píxel CUDA, 8 píxeles/vectorización SIMD)
- [ ] **Propuesta B**: descomposición por tiles 2D — evaluar impacto en localidad de caché y coalescencia GPU
- [ ] Comparar Propuesta A vs B: throughput, ocupación GPU, cache misses, complejidad
- [ ] Estrategia de pruebas: imagen referencia SEQ, comparación SIMD vs SEQ, CUDA vs SEQ, 4 tamaños de dataset

---

## 📄 PAPER IEEE (4 páginas máx.)

- [ ] Sec. 1: Motivación y contexto
- [ ] Sec. 2: Necesidades y requerimientos (funcionales, no funcionales: eficiencia energética, throughput)
- [ ] Sec. 3: Valoración de alternativas (Propuesta A vs B — criterios técnicos)
- [ ] Sec. 4: Diseño creativo (diagramas de bloques sistema completo + módulos)
- [ ] Sec. 5: Validación del diseño (resultados clave)
- [ ] Sec. 6: Análisis de resultados (comparación cuali/cuantitativa)
- [ ] Sec. 7: Conclusiones y recomendaciones
- [ ] Sec. 8: Referencias (Whitted 1980, Pharr 2023, CUDA docs, Intel Intrinsics Guide)
- [ ] Anexo: documentación del uso de IA y validación manual del código generado

---

## 🗂️ GIT

- [ ] Verificar repo privado + acceso a `rgarciatec` habilitado
- [ ] Crear ramas `feature/baseline`, `feature/simd`, `feature/cuda` desde `development`
- [ ] Merge features → development; al final: development → master + crear **tag** de entrega

---

## 📅 Fechas Clave

| Fecha | Hito |
|---|---|
| ~~14 mayo~~ | ~~Demo 1 — Propuesta y diagramas~~ ✅ |
| ~~26 mayo~~ | ~~Demo 2 — Baseline corriendo + dummies SIMD/CUDA~~ ✅ |
| 4 junio | Entrega final — Presentación 30 min + Paper PDF |
