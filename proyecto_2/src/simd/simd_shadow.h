/*
 * simd_shadow.h — Módulo 3: shadow rays vectorizados con máscara AVX2.
 * CE 4302 — Arquitectura de Computadores II.
 *
 * Función:
 *   shadow_8 — lanza 8 shadow rays (uno por lane) hacia una fuente de luz
 *              y retorna máscara de oclusión: 0xFFFFFFFF = en sombra.
 */
#pragma once
#include <immintrin.h>
#include "simd_types.h"
#include "simd_intersect.h"   /* intersect_8_sphere */
#include "light.h"            /* PointLight */
#include "scene.h"            /* Scene */

/* =======================================================================
 * MÓDULO 3 — shadow_8
 *
 * Para los 8 puntos de impacto almacenados en 'hit', construye un shadow
 * ray por lane hacia 'light' y comprueba si alguna esfera lo bloquea.
 *
 * Diferencias clave respecto a los rayos primarios:
 *   · Orígenes distintos por lane   → hit.px/py/pz (punto de impacto)
 *   · Direcciones distintas por lane → (luz − punto) normalizado
 *   · tmax_v distinto por lane       → distancia a la luz
 *
 * Offset de origen: se avanza RT_T_MIN en la dirección del shadow ray para
 * evitar auto-intersección con la esfera recién golpeada.
 *
 * Retorna:
 *   __m256 mask donde 0xFFFFFFFF = lane en sombra | 0x0 = lane iluminado
 * ======================================================================= */
inline __m256 shadow_8(const HitData8& hit,
                       const PointLight& light,
                       const Scene& scene)
{
    const __m256 eps = _mm256_set1_ps(RT_T_MIN);

    /* Vector hit_point → luz (por lane, distinto para cada punto) */
    __m256 tlx = _mm256_sub_ps(_mm256_set1_ps(light.position.x), hit.px);
    __m256 tly = _mm256_sub_ps(_mm256_set1_ps(light.position.y), hit.py);
    __m256 tlz = _mm256_sub_ps(_mm256_set1_ps(light.position.z), hit.pz);

    /* Distancia a la luz por lane: dist = |to_light| */
    __m256 dist2 = _mm256_fmadd_ps(tlx, tlx,
                   _mm256_fmadd_ps(tly, tly,
                   _mm256_mul_ps  (tlz, tlz)));
    __m256 light_dist = _mm256_sqrt_ps(dist2);

    /* Dirección normalizada: dir = to_light / dist */
    __m256 inv_d = _mm256_div_ps(_mm256_set1_ps(1.f), light_dist);
    __m256 sdx   = _mm256_mul_ps(tlx, inv_d);
    __m256 sdy   = _mm256_mul_ps(tly, inv_d);
    __m256 sdz   = _mm256_mul_ps(tlz, inv_d);

    /* Origen desplazado RT_T_MIN en la dirección del shadow ray */
    Ray8 sray;
    sray.ox = _mm256_fmadd_ps(eps, sdx, hit.px);
    sray.oy = _mm256_fmadd_ps(eps, sdy, hit.py);
    sray.oz = _mm256_fmadd_ps(eps, sdz, hit.pz);
    sray.dx = sdx;
    sray.dy = sdy;
    sray.dz = sdz;

    /* tmax_v = light_dist − RT_T_MIN: no acepta oclusores más allá de la luz */
    __m256 t_max_shad = _mm256_sub_ps(light_dist, eps);

    /* Acumular oclusión: OR de todas las esferas que bloquean el camino */
    __m256 in_shadow = _mm256_setzero_ps();
    for (const Sphere& s : scene.spheres) {
        __m256 t_dummy;
        __m256 blocked = intersect_8_sphere(
            sray,
            s.center.x, s.center.y, s.center.z, s.radius,
            t_max_shad,   /* tmax_v = distancia a la luz por lane */
            t_dummy);
        in_shadow = _mm256_or_ps(in_shadow, blocked);
    }

    return in_shadow;   /* 0xFFFFFFFF = sombra | 0x0 = iluminado */
}
