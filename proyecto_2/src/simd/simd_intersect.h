/*
 * simd_intersect.h — Módulo 2: intersección rayo-esfera y rayo-escena vectorizados.
 * CE 4302 — Arquitectura de Computadores II.
 *
 * Funciones (inline para permitir inlining completo en el compilador):
 *   intersect_8_sphere  — primitiva: 8 rayos vs 1 esfera → máscara + t
 *   intersect_8_scene   — módulo 2: 8 rayos vs escena completa → HitData8
 */
#pragma once
#include <immintrin.h>
#include "simd_types.h"
#include "scene.h"   /* Scene, Sphere */

/* =======================================================================
 * intersect_8_sphere  (primitiva interna)
 *
 * Calcula la intersección de 8 rayos contra UNA esfera usando FMA AVX2.
 *
 * Matemática (semi-discriminante):
 *   oc   = origin − center
 *   a    = D·D          (≈ 1 para rayos unitarios)
 *   hb   = oc·D         (mitad de b)
 *   c    = oc·oc − r²
 *   disc = hb² − a·c
 *   t    = (−hb ± √disc) / a   → elegir la raíz más cercana en (T_MIN, tmax_v)
 *
 * Parámetros:
 *   r8      — 8 rayos en SoA
 *   cx/cy/cz/radius — parámetros escalares de la esfera (broadcast)
 *   tmax_v  — límite superior de t por lane:
 *               · RT_T_MAX  para rayos primarios
 *               · hit.t     para closest-hit incremental
 *               · light_dist para shadow rays
 *   t_out   — [out] parámetro t del hit por lane
 *   return  — máscara de hit: 0xFFFFFFFF = hit válido | 0x0 = miss
 * ======================================================================= */
inline __m256 intersect_8_sphere(const Ray8& r8,
                                  float cx, float cy, float cz, float radius,
                                  __m256 tmax_v,
                                  __m256& t_out)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 tmin = _mm256_set1_ps(RT_T_MIN);
    const __m256 vr2  = _mm256_set1_ps(radius * radius);

    /* oc = origin − center */
    __m256 ocx = _mm256_sub_ps(r8.ox, _mm256_set1_ps(cx));
    __m256 ocy = _mm256_sub_ps(r8.oy, _mm256_set1_ps(cy));
    __m256 ocz = _mm256_sub_ps(r8.oz, _mm256_set1_ps(cz));

    /* a = D·D  (FMA) */
    __m256 a = _mm256_fmadd_ps(r8.dx, r8.dx,
               _mm256_fmadd_ps(r8.dy, r8.dy,
               _mm256_mul_ps  (r8.dz, r8.dz)));

    /* hb = oc·D  (FMA) */
    __m256 hb = _mm256_fmadd_ps(ocx, r8.dx,
                _mm256_fmadd_ps(ocy, r8.dy,
                _mm256_mul_ps  (ocz, r8.dz)));

    /* c = oc·oc − r²  (FMA) */
    __m256 oc2 = _mm256_fmadd_ps(ocx, ocx,
                 _mm256_fmadd_ps(ocy, ocy,
                 _mm256_mul_ps  (ocz, ocz)));
    __m256 c   = _mm256_sub_ps(oc2, vr2);

    /* disc = hb² − a·c */
    __m256 disc    = _mm256_sub_ps(_mm256_mul_ps(hb, hb), _mm256_mul_ps(a, c));
    __m256 disc_ok = _mm256_cmp_ps(disc, zero, _CMP_GE_OQ);  /* disc ≥ 0 */

    /* √disc — proteger de NaN: max(0, disc) antes de sqrt */
    __m256 sqd    = _mm256_sqrt_ps(_mm256_max_ps(disc, zero));
    __m256 neg_hb = _mm256_sub_ps(zero, hb);
    __m256 t_near = _mm256_div_ps(_mm256_sub_ps(neg_hb, sqd), a);
    __m256 t_far  = _mm256_div_ps(_mm256_add_ps(neg_hb, sqd), a);

    /* Validar [T_MIN, tmax_v] por lane */
    __m256 near_ok = _mm256_and_ps(_mm256_cmp_ps(t_near, tmin,  _CMP_GT_OQ),
                                   _mm256_cmp_ps(t_near, tmax_v, _CMP_LT_OQ));
    __m256 far_ok  = _mm256_and_ps(_mm256_cmp_ps(t_far,  tmin,  _CMP_GT_OQ),
                                   _mm256_cmp_ps(t_far,  tmax_v, _CMP_LT_OQ));

    /* Preferir t_near; si no es válida usar t_far */
    t_out = _mm256_blendv_ps(t_far, t_near, near_ok);

    /* Máscara final: disc ≥ 0 AND al menos una raíz en el intervalo */
    return _mm256_and_ps(disc_ok, _mm256_or_ps(near_ok, far_ok));
}

/* =======================================================================
 * MÓDULO 2 — intersect_8_scene
 *
 * Determina el impacto más cercano de cada uno de los 8 rayos contra
 * TODAS las esferas de la escena y rellena un HitData8 completo.
 *
 * Estrategia de closest-hit incremental:
 *   Se pasa hit.t como tmax_v a intersect_8_sphere → solo actualiza el lane
 *   si t_sphere < hit.t actual, garantizando siempre el mínimo sin comparación
 *   extra posterior.
 *
 * Por cada esfera que mejora el mínimo en un lane se actualizan:
 *   · hit.t                       (nuevo t mínimo)
 *   · hit.px/py/pz = O + t·D      (punto de impacto)
 *   · hit.nx/ny/nz                (normal orientada contra el rayo)
 *   · todos los campos del material
 *
 * Al terminar: hit.hit_mask = (hit.t < T_MAX) por lane.
 * ======================================================================= */
inline HitData8 intersect_8_scene(const Ray8& r8, const Scene& scene)
{
    HitData8 hit;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps( 1.f);
    const __m256 neg1 = _mm256_set1_ps(-1.f);

    /* Inicializar: sin impacto, t = T_MAX */
    hit.t        = _mm256_set1_ps(RT_T_MAX);
    hit.hit_mask = zero;
    hit.px = hit.py = hit.pz = zero;
    hit.nx = hit.ny = hit.nz = zero;
    hit.albedo_r = hit.albedo_g = hit.albedo_b = zero;
    hit.Ka = hit.Kd = hit.Ks = zero;
    hit.shininess = hit.reflectivity = zero;

    for (const Sphere& s : scene.spheres) {
        /* tmax_v = hit.t por lane: solo acepta t_sphere < mínimo actual */
        __m256 t_sphere;
        __m256 sphere_hit = intersect_8_sphere(
            r8,
            s.center.x, s.center.y, s.center.z, s.radius,
            hit.t,      /* tmax_v por lane */
            t_sphere);

        /* sphere_hit ⟹ t_sphere < hit.t  → no hace falta comparación extra */
        hit.t = _mm256_blendv_ps(hit.t, t_sphere, sphere_hit);

        /* Punto de impacto: P = O + t·D  (FMA por componente) */
        __m256 new_px = _mm256_fmadd_ps(t_sphere, r8.dx, r8.ox);
        __m256 new_py = _mm256_fmadd_ps(t_sphere, r8.dy, r8.oy);
        __m256 new_pz = _mm256_fmadd_ps(t_sphere, r8.dz, r8.oz);
        hit.px = _mm256_blendv_ps(hit.px, new_px, sphere_hit);
        hit.py = _mm256_blendv_ps(hit.py, new_py, sphere_hit);
        hit.pz = _mm256_blendv_ps(hit.pz, new_pz, sphere_hit);

        /* Normal exterior: outward = (P − center) / radius */
        __m256 inv_r  = _mm256_set1_ps(1.f / s.radius);
        __m256 out_nx = _mm256_mul_ps(_mm256_sub_ps(new_px, _mm256_set1_ps(s.center.x)), inv_r);
        __m256 out_ny = _mm256_mul_ps(_mm256_sub_ps(new_py, _mm256_set1_ps(s.center.y)), inv_r);
        __m256 out_nz = _mm256_mul_ps(_mm256_sub_ps(new_pz, _mm256_set1_ps(s.center.z)), inv_r);

        /* Orientar contra el rayo: sign = dot(D, outward) < 0 ? +1 : -1 */
        __m256 dot_d_n = _mm256_fmadd_ps(r8.dx, out_nx,
                         _mm256_fmadd_ps(r8.dy, out_ny,
                         _mm256_mul_ps  (r8.dz, out_nz)));
        __m256 front = _mm256_cmp_ps(dot_d_n, zero, _CMP_LT_OQ);
        __m256 sign  = _mm256_blendv_ps(neg1, one, front);   /* +1 cara frontal, −1 trasera */
        hit.nx = _mm256_blendv_ps(hit.nx, _mm256_mul_ps(out_nx, sign), sphere_hit);
        hit.ny = _mm256_blendv_ps(hit.ny, _mm256_mul_ps(out_ny, sign), sphere_hit);
        hit.nz = _mm256_blendv_ps(hit.nz, _mm256_mul_ps(out_nz, sign), sphere_hit);

        /* Material: broadcast escalar → blendv por lane */
        hit.albedo_r     = _mm256_blendv_ps(hit.albedo_r,     _mm256_set1_ps(s.material.albedo.x),    sphere_hit);
        hit.albedo_g     = _mm256_blendv_ps(hit.albedo_g,     _mm256_set1_ps(s.material.albedo.y),    sphere_hit);
        hit.albedo_b     = _mm256_blendv_ps(hit.albedo_b,     _mm256_set1_ps(s.material.albedo.z),    sphere_hit);
        hit.Ka           = _mm256_blendv_ps(hit.Ka,           _mm256_set1_ps(s.material.Ka),          sphere_hit);
        hit.Kd           = _mm256_blendv_ps(hit.Kd,           _mm256_set1_ps(s.material.Kd),          sphere_hit);
        hit.Ks           = _mm256_blendv_ps(hit.Ks,           _mm256_set1_ps(s.material.Ks),          sphere_hit);
        hit.shininess    = _mm256_blendv_ps(hit.shininess,    _mm256_set1_ps(s.material.shininess),   sphere_hit);
        hit.reflectivity = _mm256_blendv_ps(hit.reflectivity, _mm256_set1_ps(s.material.reflectivity),sphere_hit);
    }

    /* hit_mask: lanes donde t < T_MAX → al menos una esfera golpeada */
    hit.hit_mask = _mm256_cmp_ps(hit.t, _mm256_set1_ps(RT_T_MAX), _CMP_LT_OQ);
    return hit;
}
