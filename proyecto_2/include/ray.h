#pragma once
#include "vec3.h"

/*
 * Ray — rayo paramétrico: P(t) = origin + t * direction.
 * direction debe ser un vector unitario para que los parámetros t coincidan
 * con distancias euclidianas.
 */
struct Ray {
    Vec3 origin;
    Vec3 direction;  /* vector unitario */

    Ray() {}
    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}

    /* Retorna el punto a distancia t sobre el rayo */
    Vec3 at(float t) const { return origin + direction * t; }
};
