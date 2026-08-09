/*
 * renderer_seq.cpp — Implementación secuencial base (baseline) del ray tracer.
 *
 * Resolución objetivo: 1600 x 900
 * Algoritmo: trazado de rayos primarios + rayos de sombra + 1 nivel de reflexión.
 * Iluminación: modelo de Phong con luz ambiente, difusa y especular.
 * Salida: imagen PPM binaria (P6).
 *
 * Esta versión procesa cada píxel de manera estrictamente secuencial y sirve como
 * referencia para calcular el speedup de las versiones SIMD y CUDA.
 *
 * Compilar (Linux / GCC):
 *   g++ -std=c++17 -O2 -Wall -Iinclude src/sequential/renderer_seq.cpp -o bin/renderer_seq
 *
 * Uso:
 *   ./bin/renderer_seq [archivo_salida.ppm]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>
#include <limits>

#include "vec3.h"
#include "ray.h"
#include "hit.h"
#include "material.h"
#include "sphere.h"
#include "light.h"
#include "camera.h"
#include "scene.h"

/* ---- Constantes de la escena ---- */
static constexpr int   MAX_DEPTH = 3;      /* Rebotes máximos de reflexión */
static constexpr float T_MIN     = 1e-4f;  /* Offset para evitar auto-intersección */
static constexpr float T_MAX     = 1e30f;  /* Distancia máxima de visión */

/* -----------------------------------------------------------------------
 * intersect_scene
 * Recorre todas las esferas y retorna la intersección más cercana en (T_MIN, t_max).
 * Rellena 'rec' con los datos del impacto más próximo.
 * ----------------------------------------------------------------------- */
static bool intersect_scene(const Ray& ray,
                             const Scene& scene,
                             float t_max,
                             HitRecord& rec)
{
    HitRecord tmp;
    bool  hit_any = false;
    float closest = t_max;

    for (const Sphere& s : scene.spheres) {
        if (s.intersect(ray, T_MIN, closest, tmp)) {
            hit_any = true;
            closest = tmp.t;
            rec     = tmp;
        }
    }
    return hit_any;
}

/* -----------------------------------------------------------------------
 * in_shadow
 * Lanza un rayo de sombra desde 'origin' hacia 'light_pos'.
 * Retorna true si alguna esfera bloquea el camino antes de alcanzar la luz.
 * ----------------------------------------------------------------------- */
static bool in_shadow(const Vec3& origin,
                      const Vec3& light_pos,
                      const Scene& scene)
{
    Vec3  to_light  = light_pos - origin;
    float light_dist = to_light.length();
    Ray   shadow_ray(origin, to_light / light_dist);

    HitRecord dummy;
    for (const Sphere& s : scene.spheres) {
        /* Solo se bloquea si el obstáculo está entre el origen y la luz */
        if (s.intersect(shadow_ray, T_MIN, light_dist - T_MIN, dummy))
            return true;
    }
    return false;
}

/* -----------------------------------------------------------------------
 * phong_shade
 * Evalúa la contribución difusa y especular de Phong para una fuente de luz.
 *
 * Fórmula:
 *   diffuse  = Kd * max(0, N·L)  * Ll * albedo
 *   specular = Ks * (R·V)^shin   * Ll           (highlight blanco)
 *
 * donde:
 *   L = dirección normalizada hacia la luz
 *   R = reflect(-L, N) = dirección de reflexión de la luz
 *   V = -ray_dir       = dirección hacia el observador
 *   Ll = intensidad RGB de la fuente
 * ----------------------------------------------------------------------- */
static Vec3 phong_shade(const HitRecord& hit,
                        const PointLight& light,
                        const Vec3& ray_dir)
{
    const Material& mat = hit.material;

    Vec3  L   = (light.position - hit.point).normalize();
    Vec3  R   = Vec3::reflect(-L, hit.normal);
    Vec3  V   = -ray_dir;

    float Nd  = std::fmax(0.f, hit.normal.dot(L));
    float Ns  = std::pow(std::fmax(0.f, R.dot(V)), mat.shininess);

    Vec3 diffuse  = (mat.albedo * mat.Kd * Nd);
    Vec3 specular = Vec3(1.f, 1.f, 1.f) * (mat.Ks * Ns);  /* highlight blanco */

    return (diffuse + specular) * light.intensity;
}

/* -----------------------------------------------------------------------
 * trace
 * Traza el rayo recursivamente hasta 'depth' niveles.
 * Retorna el color calculado para ese rayo.
 * ----------------------------------------------------------------------- */
static Vec3 trace(const Ray& ray, const Scene& scene, int depth)
{
    /* Sin más rebotes: terminar con negro */
    if (depth == 0) return {0.f, 0.f, 0.f};

    HitRecord rec{};
    if (!intersect_scene(ray, scene, T_MAX, rec))
        return scene.background;

    const Material& mat = rec.material;

    /* --- Término ambiente: Ka * La * albedo --- */
    Vec3 color = mat.albedo * (scene.ambient_light * mat.Ka);

    /* --- Difuso + especular por cada fuente de luz (con prueba de sombra) --- */
    for (const PointLight& light : scene.lights) {
        if (!in_shadow(rec.point, light.position, scene))
            color += phong_shade(rec, light, ray.direction);
    }

    /* --- Reflexión especular (recursiva) --- */
    if (mat.reflectivity > 0.f) {
        Vec3 reflect_dir = Vec3::reflect(ray.direction, rec.normal).normalize();
        Ray  reflect_ray(rec.point, reflect_dir);
        Vec3 reflected_color = trace(reflect_ray, scene, depth - 1);
        color += reflected_color * mat.reflectivity;
    }

    return color.clamp01();
}

/* -----------------------------------------------------------------------
 * write_ppm
 * Escribe el framebuffer en formato PPM binario (P6) con corrección de gamma.
 * Gamma ≈ 2.0: valor_display = sqrt(valor_lineal).
 * ----------------------------------------------------------------------- */
static void write_ppm(const std::string& filename,
                      const std::vector<Vec3>& fb,
                      int w, int h)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "[SEQ] Error: no se puede abrir " << filename << "\n";
        return;
    }

    ofs << "P6\n" << w << " " << h << "\n255\n";

    for (int i = 0; i < w * h; ++i) {
        /* Corrección de gamma 2.0 + cuantización a 8 bits por canal */
        auto to_byte = [](float c) -> unsigned char {
            float clamped = std::fmax(0.f, std::fmin(1.f, c));
            return static_cast<unsigned char>(std::sqrt(clamped) * 255.f + 0.5f);
        };
        ofs << to_byte(fb[i].x) << to_byte(fb[i].y) << to_byte(fb[i].z);
    }
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    /* Uso: renderer_seq [width height [output.ppm]]
     * Ejemplo: renderer_seq 1600 900 output_seq.ppm
     * Sin argumentos: usa 1600x900 y output_seq.ppm */
    int W = 1600, H = 900;
    std::string out_file = "output_seq.ppm";
    if (argc >= 3) { W = std::stoi(argv[1]); H = std::stoi(argv[2]); }
    if (argc >= 4) out_file = argv[3];

    /* Construir escena y cámara */
    Scene  scene  = Scene::make_default();
    Camera camera = Camera::make(
        /* pos     */ {0.f, 0.5f,  3.f},
        /* look_at */ {0.f, 0.f,  -2.f},
        /* up      */ {0.f, 1.f,   0.f},
        /* vfov°   */ 50.f,
        W, H
    );

    std::vector<Vec3> framebuffer(W * H);

    /* ---- Renderizado secuencial ---- */
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            Ray  ray   = camera.get_ray(x, y);
            framebuffer[y * W + x] = trace(ray, scene, MAX_DEPTH);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[SEQ] " << W << "x" << H
              << " | depth=" << MAX_DEPTH
              << " | time: " << ms << " ms"
              << " | Mpix/s: " << (W * H / 1e6 / (ms / 1000.0))
              << "\n";

    /* ---- Escritura de imagen ---- */
    write_ppm(out_file, framebuffer, W, H);
    std::cout << "[SEQ] Imagen escrita en: " << out_file << "\n";

    return 0;
}
