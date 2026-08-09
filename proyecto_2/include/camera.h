#pragma once
#include <cmath>
#include "ray.h"

/*
 * Camera — cámara perspectiva configurada con look-at.
 *
 * Parámetros del viewport (a distancia focal = 1 en unidades de mundo):
 *   half_height = tan(vfov / 2)
 *   half_width  = aspect_ratio * half_height
 *
 * La orientación se construye con un frame ortonormal:
 *   w = normalize(pos - look_at)  (apunta hacia la cámara)
 *   u = normalize(world_up × w)   (apunta a la derecha)
 *   v = w × u                     (apunta hacia arriba)
 *
 * Convenio de píxeles: (0,0) = esquina superior-izquierda.
 */
struct Camera {
    Vec3  position;
    Vec3  lower_left;  /* Esquina inferior-izquierda del viewport en espacio mundo */
    Vec3  horizontal;  /* Vector del ancho completo del viewport */
    Vec3  vertical;    /* Vector del alto completo del viewport  */
    int   width;
    int   height;

    /*
     * Construye la cámara dados:
     *   pos      — posición del ojo
     *   look_at  — punto al que apunta la cámara
     *   world_up — vector "arriba" del mundo (normalmente (0,1,0))
     *   vfov_deg — campo de visión vertical en grados
     *   w, h     — resolución en píxeles
     */
    static Camera make(const Vec3& pos,
                       const Vec3& look_at,
                       const Vec3& world_up,
                       float vfov_deg,
                       int w, int h)
    {
        Camera cam;
        cam.width    = w;
        cam.height   = h;
        cam.position = pos;

        /* Semi-ángulos */
        float theta       = vfov_deg * (3.14159265358979f / 180.f);
        float half_h      = std::tan(theta / 2.f);
        float half_w      = (static_cast<float>(w) / static_cast<float>(h)) * half_h;

        /* Frame ortonormal de la cámara */
        Vec3 w_axis = (pos - look_at).normalize();      /* apunta hacia la cámara */
        Vec3 u_axis = world_up.cross(w_axis).normalize(); /* apunta a la derecha  */
        Vec3 v_axis = w_axis.cross(u_axis);               /* apunta hacia arriba  */

        /*
         * Esquina inferior-izquierda del viewport:
         *   se avanza 1 unidad en -w (hacia la escena),
         *   se retrocede half_w en u y half_h en v.
         */
        cam.lower_left = pos
            - w_axis
            - u_axis * half_w
            - v_axis * half_h;

        cam.horizontal = u_axis * (2.f * half_w);
        cam.vertical   = v_axis * (2.f * half_h);

        return cam;
    }

    /*
     * Genera el rayo que pasa por el centro del píxel (px, py).
     *   px en [0, width)  — columna (izquierda → derecha)
     *   py en [0, height) — fila    (arriba → abajo)
     */
    Ray get_ray(int px, int py) const {
        /* Mapea el centro del píxel a coordenadas normalizadas [0,1] */
        float u = (static_cast<float>(px) + 0.5f) / static_cast<float>(width);
        /* Invertir Y: py=0 (fila superior) → v=1 (parte alta del viewport) */
        float v = 1.f - (static_cast<float>(py) + 0.5f) / static_cast<float>(height);

        Vec3 target = lower_left + horizontal * u + vertical * v;
        Vec3 dir    = (target - position).normalize();
        return Ray(position, dir);
    }
};
