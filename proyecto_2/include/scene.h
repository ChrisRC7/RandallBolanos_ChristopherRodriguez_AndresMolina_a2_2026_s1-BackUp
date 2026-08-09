#pragma once
#include <vector>
#include "sphere.h"
#include "light.h"

/*
 * Scene — contenedor de todas las primitivas geométricas y fuentes de luz.
 * También almacena la luz ambiente global y el color del fondo (cielo).
 *
 * Escena por defecto (5 esferas + 2 luces):
 *  [0] Suelo   — esfera grande gris difusa          (centro (0,-100.5,-2), r=100)
 *  [1] Centro  — esfera roja, parcialmente reflectiva (centro (0, 0, -2),  r=0.5)
 *  [2] Izq     — esfera azul, difusa                 (centro (-1, 0, -2),  r=0.5)
 *  [3] Der     — esfera amarilla, ligeramente especular(centro ( 1, 0, -2),  r=0.5)
 *  [4] Arriba  — esfera blanca, espejo               (centro (0,  1, -2),  r=0.35)
 *
 * Luces:
 *  [0] Principal — blanca, arriba-derecha  (5, 8,  2)
 *  [1] Relleno   — azul suave, arriba-izq  (-3, 3, 1)
 */
struct Scene {
    std::vector<Sphere>     spheres;
    std::vector<PointLight> lights;
    Vec3                    ambient_light; /* Color RGB de la iluminación ambiental global */
    Vec3                    background;   /* Color de fondo cuando ningún rayo intersecta  */

    /* Construye la escena de demostración */
    static Scene make_default() {
        Scene s;
        s.ambient_light = {0.05f, 0.05f, 0.08f};
        s.background    = {0.53f, 0.73f, 0.95f};  /* azul cielo */

        /* albedo               Ka    Kd    Ks    shin  refl */

        /* [0] Suelo — esfera grande gris difusa */
        s.spheres.push_back({
            {0.f, -100.5f, -2.f}, 100.f,
            {{0.50f, 0.50f, 0.50f}, 0.1f, 0.8f, 0.1f,  8.f, 0.00f}
        });

        /* [1] Centro — rojo, ligeramente reflectiva */
        s.spheres.push_back({
            {0.f, 0.f, -2.f}, 0.5f,
            {{0.80f, 0.20f, 0.20f}, 0.1f, 0.6f, 0.4f, 64.f, 0.30f}
        });

        /* [2] Izquierda — azul, difusa */
        s.spheres.push_back({
            {-1.0f, 0.f, -2.f}, 0.5f,
            {{0.20f, 0.20f, 0.80f}, 0.1f, 0.8f, 0.2f, 16.f, 0.00f}
        });

        /* [3] Derecha — amarillo, ligeramente especular */
        s.spheres.push_back({
            {1.0f, 0.f, -2.f}, 0.5f,
            {{0.80f, 0.80f, 0.20f}, 0.1f, 0.7f, 0.3f, 32.f, 0.00f}
        });

        /* [4] Arriba — blanca/plateada, espejo */
        s.spheres.push_back({
            {0.f, 1.0f, -2.f}, 0.35f,
            {{0.90f, 0.90f, 0.90f}, 0.05f, 0.1f, 0.9f, 128.f, 0.90f}
        });

        /* Luz principal: blanca intensa, arriba a la derecha */
        s.lights.push_back({{ 5.f,  8.f, 2.f}, {1.0f, 1.0f, 1.0f}});

        /* Luz de relleno: azul suave, arriba a la izquierda */
        s.lights.push_back({{-3.f,  3.f, 1.f}, {0.3f, 0.3f, 0.5f}});

        return s;
    }
};
