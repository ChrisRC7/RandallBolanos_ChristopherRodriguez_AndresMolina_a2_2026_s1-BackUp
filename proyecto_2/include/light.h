#pragma once
#include "vec3.h"

/*
 * PointLight — fuente de luz puntual en el espacio mundo.
 * La intensidad es un vector RGB para permitir luces con color.
 */
struct PointLight {
    Vec3 position;   /* Posición de la fuente en espacio mundo */
    Vec3 intensity;  /* Intensidad RGB (color × brillo) */
};
