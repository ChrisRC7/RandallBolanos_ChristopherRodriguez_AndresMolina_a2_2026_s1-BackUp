#pragma once
#include "vec3.h"

/*
 * Material — propiedades de superficie para el modelo de iluminación de Phong.
 *
 * color_phong = Ka * La * albedo
 *             + Kd * max(0, N·L) * Ll * albedo      (difuso)
 *             + Ks * (R·V)^shininess * Ll            (especular, highlight blanco)
 *
 * donde La = luz ambiente, Ll = intensidad de la fuente puntual.
 */
struct Material {
    Vec3  albedo;       /* Color base de la superficie (RGB en [0,1]) */
    float Ka;           /* Coeficiente de reflexión ambiental  [0,1] */
    float Kd;           /* Coeficiente de reflexión difusa     [0,1] */
    float Ks;           /* Coeficiente de reflexión especular  [0,1] */
    float shininess;    /* Exponente de Phong (mayor = highlight más angosto) */
    float reflectivity; /* Fracción de color aportada por el rayo reflejado [0,1] */
};
