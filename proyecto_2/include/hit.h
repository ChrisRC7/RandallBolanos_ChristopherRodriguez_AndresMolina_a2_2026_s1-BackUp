#pragma once
#include "vec3.h"
#include "material.h"

/*
 * HitRecord — datos registrados cuando un rayo intersecta una superficie.
 * Se rellena por la función de intersección de cada primitiva geométrica.
 */
struct HitRecord {
    Vec3     point;      /* Posición en espacio mundo del punto de intersección  */
    Vec3     normal;     /* Normal unitaria en el punto; siempre apunta contra el rayo */
    Material material;   /* Material de la superficie golpeada                  */
    float    t;          /* Parámetro del rayo: point = ray.at(t)               */
    bool     front_face; /* true si el rayo golpeó la cara exterior de la esfera */
};
