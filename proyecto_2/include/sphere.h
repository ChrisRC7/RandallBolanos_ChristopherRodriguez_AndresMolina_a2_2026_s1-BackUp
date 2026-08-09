#pragma once
#include <cmath>
#include "ray.h"
#include "hit.h"

/*
 * Sphere — primitiva geométrica esférica definida por centro, radio y material.
 *
 * La intersección rayo-esfera resuelve la ecuación cuadrática:
 *   |O + t·D - C|² = r²
 *   (D·D)t² + 2(D·(O-C))t + ((O-C)·(O-C) - r²) = 0
 *
 * Se usa la forma con semi-discriminante para reducir cancelación numérica:
 *   a  = D·D  (= 1 si D es unitario)
 *   hb = D·(O-C)           (mitad de b)
 *   c  = (O-C)·(O-C) - r²
 *   disc = hb² - a·c
 */
struct Sphere {
    Vec3     center;
    float    radius;
    Material material;

    /*
     * Verifica si el rayo 'ray' intersecta la esfera en el intervalo [t_min, t_max].
     * Si hay intersección, rellena 'rec' con los datos del impacto y retorna true.
     * Se elige siempre la raíz más pequeña (impacto más cercano al origen).
     */
    bool intersect(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
        Vec3  oc  = ray.origin - center;
        float a   = ray.direction.length_sq();
        float hb  = oc.dot(ray.direction);
        float c   = oc.length_sq() - radius * radius;
        float disc = hb * hb - a * c;

        if (disc < 0.f) return false;

        float sqrt_disc = std::sqrt(disc);

        /* Probar primero la raíz negativa (más cercana) */
        float t = (-hb - sqrt_disc) / a;
        if (t < t_min || t > t_max) {
            /* Probar la raíz positiva (impacto desde adentro) */
            t = (-hb + sqrt_disc) / a;
            if (t < t_min || t > t_max) return false;
        }

        rec.t     = t;
        rec.point = ray.at(t);

        /* Normal exterior; se orienta para que siempre sea contra el rayo */
        Vec3 outward_normal  = (rec.point - center) / radius;
        rec.front_face       = ray.direction.dot(outward_normal) < 0.f;
        rec.normal           = rec.front_face ? outward_normal : -outward_normal;
        rec.material         = material;

        return true;
    }
};
