/*
 * simd_trace.h — Módulo 5: pipeline completo de tracing con reflexión (AVX2).
 * CE 4302 — Arquitectura de Computadores II.
 *
 * Función:
 *   trace_8 — traza 8 rayos con Phong + shadow + reflexión iterativa
 *             hasta RT_MAX_DEPTH rebotes.
 *
 * Estrategia iterativa (equivalente a la recursión del SEQ):
 *
 *   SEQ (recursivo):
 *     trace(ray, depth):
 *       if no hit  → return background
 *       color = ambient + sum(phong * not_in_shadow)
 *       if reflective: color += reflectivity * trace(reflect_ray, depth-1)
 *       return clamp(color)
 *
 *   SIMD (iterativo con máscara de lanes activos):
 *     active = todos los lanes | weight = 1.0 por lane
 *     for d in [0, RT_MAX_DEPTH):
 *       hit = intersect_8_scene(cur_ray)
 *       miss lanes → color += background * weight; desactivar
 *       hit lanes  → color += (ambient + phong) * weight
 *       lanes con reflectivity > 0 → preparar rayo reflejado,
 *                                     weight *= reflectivity
 *       el resto   → desactivar (contribución completa)
 *     color = clamp(color, 0, 1)
 *
 * Diferencia menor respecto al SEQ: el SEQ aplica clamp01() en CADA nivel
 * de recursión; aquí se aplica solo al final. Para la mayoría de píxeles
 * el resultado es idéntico; para superficies muy reflectivas puede diferir
 * en ~1e-3. La función de validación (módulo 7) cuantifica esto.
 */
#pragma once
#include <immintrin.h>
#include "simd_types.h"
#include "simd_intersect.h"   /* intersect_8_scene  */
#include "simd_shadow.h"      /* shadow_8           */
#include "simd_shade.h"       /* phong_8            */
#include "scene.h"

/* =======================================================================
 * MÓDULO 5 — trace_8
 *
 * Parámetros:
 *   r8_in   — 8 rayos primarios (Ray8 en SoA)
 *   scene   — escena completa (esferas, luces, ambiente, fondo)
 *
 * Salidas:
 *   col_r / col_g / col_b — color final por lane, en [0, 1]
 * ======================================================================= */
inline void trace_8(const Ray8& r8_in,
                    const Scene& scene,
                    __m256& col_r, __m256& col_g, __m256& col_b)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.f);

    /* Color acumulado y peso por lane */
    col_r = col_g = col_b = zero;
    __m256 weight = one;   /* cuánto contribuye el rayo actual al píxel */

    /* Máscara de lanes activos: 0xFFFFFFFF = sigue siendo trazado */
    __m256 active = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

    /* Rayo actual (se actualiza en cada rebote) */
    Ray8 cur = r8_in;

    for (int d = 0; d < RT_MAX_DEPTH; ++d) {

        /* ── INTERSECCIÓN: closest hit contra toda la escena ── */
        HitData8 hit = intersect_8_scene(cur, scene);

        /* ── MISS lanes: añadir fondo * weight y desactivar ──
         *   andnot(hit_mask, active) = active AND NOT hit_mask = miss_active
         *
         *   IMPORTANTE: miss es una máscara de bits (0xFFFFFFFF o 0).
         *   Se usa _mm256_and_ps(miss, valor) para enmascarar el float — NO
         *   como operando aritmético directo (0xFFFFFFFF = NaN en IEEE 754). */
        __m256 miss = _mm256_andnot_ps(hit.hit_mask, active);
        col_r = _mm256_add_ps(col_r, _mm256_and_ps(miss, _mm256_mul_ps(_mm256_set1_ps(scene.background.x), weight)));
        col_g = _mm256_add_ps(col_g, _mm256_and_ps(miss, _mm256_mul_ps(_mm256_set1_ps(scene.background.y), weight)));
        col_b = _mm256_add_ps(col_b, _mm256_and_ps(miss, _mm256_mul_ps(_mm256_set1_ps(scene.background.z), weight)));

        /* Desactivar lanes que no golpearon nada */
        active = _mm256_and_ps(active, hit.hit_mask);

        /* Salida temprana: todos los lanes terminaron */
        if (_mm256_movemask_ps(active) == 0) break;

        /* ── TÉRMINO AMBIENTE: Ka * ambient_light * albedo * weight ──
         *   Solo para lanes activos (and con active). */
        __m256 amb_w = _mm256_mul_ps(hit.Ka, weight);
        __m256 amb_r = _mm256_and_ps(active, _mm256_mul_ps(amb_w,
                           _mm256_mul_ps(_mm256_set1_ps(scene.ambient_light.x), hit.albedo_r)));
        __m256 amb_g = _mm256_and_ps(active, _mm256_mul_ps(amb_w,
                           _mm256_mul_ps(_mm256_set1_ps(scene.ambient_light.y), hit.albedo_g)));
        __m256 amb_b = _mm256_and_ps(active, _mm256_mul_ps(amb_w,
                           _mm256_mul_ps(_mm256_set1_ps(scene.ambient_light.z), hit.albedo_b)));
        col_r = _mm256_add_ps(col_r, amb_r);
        col_g = _mm256_add_ps(col_g, amb_g);
        col_b = _mm256_add_ps(col_b, amb_b);

        /* ── DIFUSO + ESPECULAR por cada fuente de luz ──
         *   lit_mask = active AND NOT in_shadow */
        for (const PointLight& light : scene.lights) {
            __m256 shadow = shadow_8(hit, light, scene);
            __m256 lit    = _mm256_andnot_ps(shadow, active);   /* ~shadow & active */

            __m256 ph_r, ph_g, ph_b;
            phong_8(hit, cur.dx, cur.dy, cur.dz, light, lit, ph_r, ph_g, ph_b);

            /* Escalar contribución Phong por weight y acumular */
            col_r = _mm256_fmadd_ps(weight, ph_r, col_r);
            col_g = _mm256_fmadd_ps(weight, ph_g, col_g);
            col_b = _mm256_fmadd_ps(weight, ph_b, col_b);
        }

        /* ── REFLEXIÓN: preparar siguiente rayo para lanes reflectivos ──
         *   El último nivel (d == RT_MAX_DEPTH-1) equivale al depth=0 del SEQ
         *   que retorna {0,0,0} → no hace falta preparar el rayo. */
        if (d < RT_MAX_DEPTH - 1) {
            /* Lanes que continúan: activos Y con reflectivity > 0 */
            __m256 has_refl = _mm256_and_ps(active,
                                  _mm256_cmp_ps(hit.reflectivity, zero, _CMP_GT_OQ));

            /* Actualizar weight para el siguiente rebote */
            weight = _mm256_blendv_ps(weight,
                         _mm256_mul_ps(weight, hit.reflectivity),
                         has_refl);

            /* Solo los lanes reflectivos siguen activos */
            active = has_refl;
            if (_mm256_movemask_ps(active) == 0) break;

            /* ── Calcular dirección reflejada: R = D − 2*(D·N)*N ──
             *   Equivale a Vec3::reflect(ray.direction, rec.normal) del SEQ.
             *   R es unitario si D y N son unitarios (no requiere renormalizar). */
            __m256 DdN    = _mm256_fmadd_ps(cur.dx, hit.nx,
                            _mm256_fmadd_ps(cur.dy, hit.ny,
                            _mm256_mul_ps  (cur.dz, hit.nz)));
            __m256 two_dn = _mm256_add_ps(DdN, DdN);  /* 2*(D·N) */
            /* refl_d = D − 2*(D·N)*N = fnmadd(two_dn, N, D) = D - two_dn*N */
            __m256 rdx = _mm256_fnmadd_ps(two_dn, hit.nx, cur.dx);
            __m256 rdy = _mm256_fnmadd_ps(two_dn, hit.ny, cur.dy);
            __m256 rdz = _mm256_fnmadd_ps(two_dn, hit.nz, cur.dz);

            /* Normalizar dirección reflejada (robustez numérica) */
            __m256 inv_rlen = _mm256_div_ps(one,
                                  _mm256_sqrt_ps(
                                      _mm256_fmadd_ps(rdx, rdx,
                                      _mm256_fmadd_ps(rdy, rdy,
                                      _mm256_mul_ps  (rdz, rdz)))));
            rdx = _mm256_mul_ps(rdx, inv_rlen);
            rdy = _mm256_mul_ps(rdy, inv_rlen);
            rdz = _mm256_mul_ps(rdz, inv_rlen);

            /* Nuevo origen: hit.point + RT_T_MIN * reflect_dir
             * (offset para evitar auto-intersección, igual que SEQ) */
            __m256 eps = _mm256_set1_ps(RT_T_MIN);
            cur.ox = _mm256_fmadd_ps(eps, rdx, hit.px);
            cur.oy = _mm256_fmadd_ps(eps, rdy, hit.py);
            cur.oz = _mm256_fmadd_ps(eps, rdz, hit.pz);
            cur.dx = rdx;
            cur.dy = rdy;
            cur.dz = rdz;
        }
    }

    /* ── Clamp final a [0, 1] por canal ── */
    col_r = _mm256_min_ps(one, _mm256_max_ps(zero, col_r));
    col_g = _mm256_min_ps(one, _mm256_max_ps(zero, col_g));
    col_b = _mm256_min_ps(one, _mm256_max_ps(zero, col_b));
}
