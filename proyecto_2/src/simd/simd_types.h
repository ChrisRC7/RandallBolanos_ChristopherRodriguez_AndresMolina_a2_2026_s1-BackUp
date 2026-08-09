/*
 * simd_types.h — Tipos y constantes compartidas del pipeline SIMD/AVX2.
 * CE 4302 — Arquitectura de Computadores II.
 *
 * Contiene:
 *   — Constantes del ray tracer (T_MIN, T_MAX, MAX_DEPTH)
 *   — Ray8     : 8 rayos empaquetados (Structure-of-Arrays)
 *   — HitData8 : resultado del closest hit para 8 rayos simultáneos
 */
#pragma once
#include <immintrin.h>  /* __m256, intrínsecos AVX2 */

/* ---- Constantes del ray tracer ---- */
static constexpr float RT_T_MIN     = 1e-4f;
static constexpr float RT_T_MAX     = 1e30f;
static constexpr int   RT_MAX_DEPTH = 3;      /* Igual que SEQ */

/* =======================================================================
 * Ray8 — Ocho rayos empaquetados en Structure-of-Arrays (SoA).
 *
 * Cada campo __m256 contiene el mismo componente de los 8 rayos:
 *   ox[0..7] = x del origen del rayo 0..7
 *   dx[0..7] = x de la dirección del rayo 0..7
 *   ... etc.
 *
 * SoA vs AoS: SoA permite cargar/operar los 8 valores del mismo
 * componente con una sola instrucción SIMD (más eficiente en AVX2).
 * ======================================================================= */
struct Ray8 {
    __m256 ox, oy, oz;  /* Orígenes: pueden ser distintos (shadow rays) o iguales (primarios) */
    __m256 dx, dy, dz;  /* Direcciones unitarias, distintas por píxel                         */
};

/* =======================================================================
 * HitData8 — Impacto más cercano para cada uno de los 8 rayos.
 *
 * Todos los campos son __m256 (un float por lane = un valor por rayo).
 * Se rellena por intersect_8_scene() y se consume por phong_8()/trace_8().
 *
 * Layout (16 registros YMM):
 *   Geometría    px  py  pz   — punto de impacto en espacio mundo (P = O + t·D)
 *                nx  ny  nz   — normal unitaria orientada contra el rayo
 *   Material     albedo_r/g/b — color base (difuso)
 *                Ka  Kd  Ks   — coeficientes de Phong
 *                shininess    — exponente especular (mayor = highlight más angosto)
 *                reflectivity — fracción de color aportada por reflexión
 *   Control      t            — parámetro t del impacto (distancia)
 *                hit_mask     — 0xFFFFFFFF si lane golpeó algo; 0x0 si no
 * ======================================================================= */
struct HitData8 {
    /* ---- Geometría ---- */
    __m256 px, py, pz;                    /* Posición del impacto  */
    __m256 nx, ny, nz;                    /* Normal orientada       */

    /* ---- Material (Phong) ---- */
    __m256 albedo_r, albedo_g, albedo_b;  /* Color base RGB         */
    __m256 Ka;                            /* Coef. ambiental        */
    __m256 Kd;                            /* Coef. difuso           */
    __m256 Ks;                            /* Coef. especular        */
    __m256 shininess;                     /* Exponente de Phong     */
    __m256 reflectivity;                  /* Fracción de reflexión  */

    /* ---- Control ---- */
    __m256 t;                             /* Distancia al impacto   */
    __m256 hit_mask;                      /* Máscara de hit válido  */
};
