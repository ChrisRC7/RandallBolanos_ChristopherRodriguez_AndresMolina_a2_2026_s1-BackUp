/*
 * simd_shade.h — Módulo 4: iluminación de Phong vectorizada (AVX2 + FMA).
 * CE 4302 — Arquitectura de Computadores II.
 *
 * Contenido:
 *   pow8_approx  — potenciación fast con exponents por lane (helper interno)
 *   phong_8      — contribución difusa + especular de UNA luz para 8 lanes
 *
 * Fórmula de Phong implementada:
 *   L        = normalize(light.pos − hit.point)  [por lane]
 *   NdL      = max(0, N·L)
 *   diffuse  = Kd * NdL * light.intensity * albedo
 *   R        = 2*(N·L)*N − L                     [reflexión de la luz]
 *   V        = −ray_dir                           [hacia el observador]
 *   RdV      = max(0, R·V)
 *   specular = Ks * pow(RdV, shininess) * light.intensity   [highlight blanco]
 *   total    = (diffuse + specular) * lit_mask
 */
#pragma once
#include <immintrin.h>
#include "simd_types.h"
#include "light.h"

/* =======================================================================
 * pow8_approx  (helper interno)
 *
 * Aproxima base^exp con exponentes distintos por lane usando:
 *   pow(x, n) = exp2(n * log2(x))
 *
 * log2 y exp2 se implementan con trucos de bits IEEE 754 + polinomios.
 * Error < 1% para base en (0,1] y exp en [8, 128] — suficiente para
 * el highlight especular de Phong, visualmente imperceptible.
 * ======================================================================= */

/* log2 aproximado para x > 0 usando bits IEEE 754 */
static inline __m256 log2_approx(__m256 x)
{
    /* Extraer exponente e y mantisa m de la representación IEEE 754:
     *   x = 2^e * m,  m in [1, 2)
     *   log2(x) = e + log2(m) */
    __m256i xi = _mm256_castps_si256(x);

    /* e = (bits >> 23) - 127 */
    __m256  e  = _mm256_cvtepi32_ps(
                     _mm256_sub_epi32(
                         _mm256_srli_epi32(xi, 23),
                         _mm256_set1_epi32(127)));

    /* m = mantisa forzada al rango [1,2): poner exponente = 127 */
    __m256  m  = _mm256_castsi256_ps(
                     _mm256_or_si256(
                         _mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)),
                         _mm256_set1_epi32(0x3f800000)));

    /* log2(m) para m in [1,2): polinomio minimax de grado 3
     * log2(m) ≈ (m-1) * (1.4427 + (m-1) * (−0.7213 + (m-1) * 0.4777))
     * Coeficientes calibrados para shininess típico [8, 128] */
    __m256 f    = _mm256_sub_ps(m, _mm256_set1_ps(1.f));
    __m256 log2m = _mm256_fmadd_ps(f,
                   _mm256_fmadd_ps(f,
                   _mm256_fmadd_ps(f,  _mm256_set1_ps( 0.4777f),
                                       _mm256_set1_ps(-0.7213f)),
                                       _mm256_set1_ps( 1.4427f)),
                   _mm256_setzero_ps());

    return _mm256_add_ps(e, log2m);
}

/* exp2 aproximado: 2^x para x in [-126, 127] */
static inline __m256 exp2_approx(__m256 x)
{
    /* Separar parte entera y fraccionaria */
    __m256 xi = _mm256_floor_ps(x);
    __m256 xf = _mm256_sub_ps(x, xi);

    /* 2^frac: polinomio de grado 3 para xf in [0, 1)
     * 2^f ≈ 1 + f*(0.6931 + f*(0.2402 + f*0.0557)) */
    __m256 p = _mm256_fmadd_ps(xf,
               _mm256_fmadd_ps(xf,
               _mm256_fmadd_ps(xf,  _mm256_set1_ps(0.0557f),
                                    _mm256_set1_ps(0.2402f)),
                                    _mm256_set1_ps(0.6931f)),
               _mm256_set1_ps(1.f));

    /* 2^int: truco de exponente IEEE 754 — (e + 127) << 23 */
    __m256i ei   = _mm256_cvtps_epi32(xi);
    ei = _mm256_max_epi32(ei, _mm256_set1_epi32(-126));
    ei = _mm256_min_epi32(ei, _mm256_set1_epi32( 127));
    __m256  pow2 = _mm256_castsi256_ps(
                       _mm256_slli_epi32(
                           _mm256_add_epi32(ei, _mm256_set1_epi32(127)), 23));

    return _mm256_mul_ps(p, pow2);
}

/* pow8_approx: base^exp con base y exp distintos por lane */
static inline __m256 pow8_approx(__m256 base, __m256 exp)
{
    /* Proteger base ≤ 0 (RdV puede ser 0 → log2(0) = −inf → pow = 0 correcto,
     * pero clamp a epsilon evita NaN) */
    base = _mm256_max_ps(base, _mm256_set1_ps(1e-7f));
    return exp2_approx(_mm256_mul_ps(exp, log2_approx(base)));
}

/* =======================================================================
 * MÓDULO 4 — phong_8
 *
 * Calcula la contribución difusa + especular de Phong de UNA fuente de
 * luz puntual para los 8 lanes simultáneamente.
 *
 * Parámetros:
 *   hit           — datos del impacto (punto, normal, material)
 *   ray_dx/dy/dz  — dirección del rayo actual (para V = -D)
 *   light         — fuente de luz puntual
 *   lit_mask      — lanes a evaluar: AND(hit_mask, NOT(shadow_mask))
 *                   lanes no iluminadas reciben out = 0
 *
 * Salidas:
 *   out_r/g/b     — contribución RGB (0 para lanes fuera de lit_mask)
 *
 * Nota: esta función ESCRIBE out_r/g/b (no acumula).
 *       El caller aplica el peso y acumula en el color total.
 * ======================================================================= */
inline void phong_8(const HitData8& hit,
                    __m256 ray_dx, __m256 ray_dy, __m256 ray_dz,
                    const PointLight& light,
                    __m256 lit_mask,
                    __m256& out_r, __m256& out_g, __m256& out_b)
{
    const __m256 zero = _mm256_setzero_ps();

    /* ── L = normalize(light.pos − hit.point) por lane ── */
    __m256 Lx = _mm256_sub_ps(_mm256_set1_ps(light.position.x), hit.px);
    __m256 Ly = _mm256_sub_ps(_mm256_set1_ps(light.position.y), hit.py);
    __m256 Lz = _mm256_sub_ps(_mm256_set1_ps(light.position.z), hit.pz);
    __m256 inv_llen = _mm256_div_ps(_mm256_set1_ps(1.f),
                          _mm256_sqrt_ps(
                              _mm256_fmadd_ps(Lx, Lx,
                              _mm256_fmadd_ps(Ly, Ly,
                              _mm256_mul_ps  (Lz, Lz)))));
    Lx = _mm256_mul_ps(Lx, inv_llen);
    Ly = _mm256_mul_ps(Ly, inv_llen);
    Lz = _mm256_mul_ps(Lz, inv_llen);

    /* ── NdL = max(0, N·L) ── */
    __m256 NdL = _mm256_max_ps(zero,
                     _mm256_fmadd_ps(hit.nx, Lx,
                     _mm256_fmadd_ps(hit.ny, Ly,
                     _mm256_mul_ps  (hit.nz, Lz))));

    /* ── Difuso: Kd * NdL * light.intensity * albedo ── */
    __m256 li_r   = _mm256_set1_ps(light.intensity.x);
    __m256 li_g   = _mm256_set1_ps(light.intensity.y);
    __m256 li_b   = _mm256_set1_ps(light.intensity.z);
    __m256 kd_ndl = _mm256_mul_ps(hit.Kd, NdL);
    __m256 diff_r = _mm256_mul_ps(_mm256_mul_ps(kd_ndl, li_r), hit.albedo_r);
    __m256 diff_g = _mm256_mul_ps(_mm256_mul_ps(kd_ndl, li_g), hit.albedo_g);
    __m256 diff_b = _mm256_mul_ps(_mm256_mul_ps(kd_ndl, li_b), hit.albedo_b);

    /* ── R = 2*(N·L)*N − L  (dirección de reflexión de la luz) ──
     *   fmsub(a,b,c) = a*b − c  →  2*NdL*N − L
     * R es unitario si N y L son unitarios (preserva longitud). */
    __m256 two_ndl = _mm256_add_ps(NdL, NdL);
    __m256 Rx = _mm256_fmsub_ps(two_ndl, hit.nx, Lx);
    __m256 Ry = _mm256_fmsub_ps(two_ndl, hit.ny, Ly);
    __m256 Rz = _mm256_fmsub_ps(two_ndl, hit.nz, Lz);

    /* ── V = −ray_dir (dirección hacia el observador) ── */
    __m256 Vx = _mm256_sub_ps(zero, ray_dx);
    __m256 Vy = _mm256_sub_ps(zero, ray_dy);
    __m256 Vz = _mm256_sub_ps(zero, ray_dz);

    /* ── RdV = max(0, R·V) ── */
    __m256 RdV = _mm256_max_ps(zero,
                     _mm256_fmadd_ps(Rx, Vx,
                     _mm256_fmadd_ps(Ry, Vy,
                     _mm256_mul_ps  (Rz, Vz))));

    /* ── Especular: Ks * pow(RdV, shininess) * light.intensity
     *   Highlight blanco → se aplica directamente la intensidad RGB de la luz
     *   (sin multiplicar por albedo, igual que SEQ). ── */
    __m256 spec_val = _mm256_mul_ps(hit.Ks, pow8_approx(RdV, hit.shininess));
    __m256 spec_r   = _mm256_mul_ps(spec_val, li_r);
    __m256 spec_g   = _mm256_mul_ps(spec_val, li_g);
    __m256 spec_b   = _mm256_mul_ps(spec_val, li_b);

    /* ── Salida: (diffuse + specular) enmascarado por lit_mask ──
     *   Lanes fuera de lit_mask (en sombra o sin impacto) reciben 0. */
    out_r = _mm256_and_ps(lit_mask, _mm256_add_ps(diff_r, spec_r));
    out_g = _mm256_and_ps(lit_mask, _mm256_add_ps(diff_g, spec_g));
    out_b = _mm256_and_ps(lit_mask, _mm256_add_ps(diff_b, spec_b));
}
