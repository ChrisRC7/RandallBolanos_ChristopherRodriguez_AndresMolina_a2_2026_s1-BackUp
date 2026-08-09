/*
 * renderer_simd_dummy.cpp — Pipeline SIMD/AVX2 del ray tracer (entrada principal).
 * CE 4302 — Arquitectura de Computadores II.
 *
 * Este archivo solo contiene:
 *   generate_8_rays  — empaqueta 8 rayos primarios en un Ray8
 *   write_ppm        — escribe el framebuffer a disco
 *   main             — bucle de render + scalar fallback
 *
 * Toda la lógica del pipeline está en los headers modulares:
 *   simd_types.h      — Ray8, HitData8, constantes
 *   simd_intersect.h  — intersect_8_sphere, intersect_8_scene  (módulo 2)
 *   simd_shadow.h     — shadow_8                               (módulo 3)
 *   simd_shade.h      — phong_8                                (módulo 4, próximo)
 *   simd_trace.h      — trace_8                                (módulo 5, próximo)
 *
 * Compilar (Linux):
 *   g++ -std=c++17 -O3 -march=native -mavx2 -mfma -Wall \
 *       -I include -I src/simd \
 *       src/simd/renderer_simd_dummy.cpp -o bin/renderer_simd_dummy
 */

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

/* ---- Módulos SIMD ---- */
#include "simd_types.h"       /* Ray8, HitData8, RT_T_MIN/MAX/MAX_DEPTH */
#include "simd_intersect.h"   /* intersect_8_sphere, intersect_8_scene  */
#include "simd_shadow.h"      /* shadow_8                                */
#include "simd_shade.h"       /* phong_8, pow8_approx                    */
#include "simd_trace.h"       /* trace_8                                 */

/* ---- Headers del proyecto ---- */
#include "vec3.h"
#include "ray.h"
#include "hit.h"
#include "material.h"
#include "sphere.h"
#include "light.h"
#include "camera.h"
#include "scene.h"

/* =======================================================================
 * Módulo 6 — Scalar fallback (Phong completo, idéntico al SEQ)
 *
 * Se usa para los píxeles residuales cuando WIDTH no es múltiplo de 8.
 * Las tres funciones siguientes son copia directa de renderer_seq.cpp.
 * ======================================================================= */
static bool sc_intersect(const Ray& ray, const Scene& scene, float t_max,
                          HitRecord& rec)
{
    HitRecord tmp; bool hit_any = false; float closest = t_max;
    for (const Sphere& s : scene.spheres)
        if (s.intersect(ray, RT_T_MIN, closest, tmp)) {
            hit_any = true; closest = tmp.t; rec = tmp; }
    return hit_any;
}

static bool sc_in_shadow(const Vec3& origin, const Vec3& light_pos,
                          const Scene& scene)
{
    Vec3  to_l = light_pos - origin;
    float dist = to_l.length();
    Ray   sr(origin, to_l / dist);
    HitRecord dummy;
    for (const Sphere& s : scene.spheres)
        if (s.intersect(sr, RT_T_MIN, dist - RT_T_MIN, dummy)) return true;
    return false;
}

static Vec3 sc_phong(const HitRecord& hit, const PointLight& light,
                     const Vec3& ray_dir)
{
    const Material& mat = hit.material;
    Vec3  L  = (light.position - hit.point).normalize();
    Vec3  R  = Vec3::reflect(-L, hit.normal);
    Vec3  V  = -ray_dir;
    float Nd = std::fmax(0.f, hit.normal.dot(L));
    float Ns = std::pow(std::fmax(0.f, R.dot(V)), mat.shininess);
    return (mat.albedo * (mat.Kd * Nd) + Vec3(1.f,1.f,1.f) * (mat.Ks * Ns))
           * light.intensity;
}

static Vec3 sc_trace(const Ray& ray, const Scene& scene, int depth)
{
    if (depth == 0) return {0.f, 0.f, 0.f};
    HitRecord rec{};
    if (!sc_intersect(ray, scene, RT_T_MAX, rec)) return scene.background;
    const Material& mat = rec.material;
    Vec3 color = mat.albedo * (scene.ambient_light * mat.Ka);
    for (const PointLight& light : scene.lights)
        if (!sc_in_shadow(rec.point, light.position, scene))
            color += sc_phong(rec, light, ray.direction);
    if (mat.reflectivity > 0.f) {
        Vec3 rdir = Vec3::reflect(ray.direction, rec.normal).normalize();
        color += sc_trace(Ray(rec.point, rdir), scene, depth - 1)
                 * mat.reflectivity;
    }
    return color.clamp01();
}

/* =======================================================================
 * Módulo 7 — validate_vs_seq
 * ======================================================================= */
/* forward declaration — definición completa más abajo */
static Ray8 generate_8_rays(const Camera& cam, int x, int y);

/*
 * Renderiza la escena con el pipeline SIMD y con el scalar completo (SEQ),
 * luego compara canal a canal con tolerancia 'tol' (por defecto 1e-4).
 *
 * Salida por consola:
 *   max_diff, mean_diff, canales fuera de tolerancia, PASS / FAIL
 *
 * Nota: la comparación se hace en espacio lineal [0,1] antes de gamma.
 */
static void validate_vs_seq(int W, int H, float tol = 1e-4f)
{
    Scene  scene  = Scene::make_default();
    Camera camera = Camera::make(
        {0.f, 0.5f,  3.f},
        {0.f, 0.f,  -2.f},
        {0.f, 1.f,   0.f},
        50.f, W, H);

    const int N  = W * H;
    const int x8 = (W / 8) * 8;

    /* ---- Framebuffer SIMD ---- */
    std::vector<Vec3> fb_simd(N);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < x8; x += 8) {
            Ray8 rays = generate_8_rays(camera, x, y);
            __m256 cr, cg, cb;
            trace_8(rays, scene, cr, cg, cb);
            alignas(32) float ro[8], go[8], bo[8];
            _mm256_store_ps(ro, cr); _mm256_store_ps(go, cg); _mm256_store_ps(bo, cb);
            for (int i = 0; i < 8; ++i)
                fb_simd[y*W+x+i] = {ro[i], go[i], bo[i]};
        }
        for (int x = x8; x < W; ++x)
            fb_simd[y*W+x] = sc_trace(camera.get_ray(x,y), scene, RT_MAX_DEPTH);
    }

    /* ---- Framebuffer SEQ (scalar completo) ---- */
    std::vector<Vec3> fb_seq(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            fb_seq[y*W+x] = sc_trace(camera.get_ray(x,y), scene, RT_MAX_DEPTH);

    /* ---- Comparar canal a canal ---- */
    float max_diff  = 0.f;
    double sum_diff = 0.0;
    int    over_tol = 0;
    for (int i = 0; i < N; ++i) {
        float dr = std::fabs(fb_simd[i].x - fb_seq[i].x);
        float dg = std::fabs(fb_simd[i].y - fb_seq[i].y);
        float db = std::fabs(fb_simd[i].z - fb_seq[i].z);
        max_diff  = std::fmax(max_diff, std::fmax(dr, std::fmax(dg, db)));
        sum_diff += dr + dg + db;
        if (dr > tol || dg > tol || db > tol) ++over_tol;
    }
    double mean_diff = sum_diff / (3.0 * N);

    std::cout << "\n[VALIDATE] " << W << "x" << H << " | tol=" << tol << "\n"
              << "  max_diff  : " << max_diff  << "\n"
              << "  mean_diff : " << mean_diff << "\n"
              << "  pixeles > tol: " << over_tol << " / " << N << "\n"
              << "  " << (over_tol == 0 ? "PASS ✓" : "FAIL ✗  (tolerancia superada)") << "\n\n";
}

/* -----------------------------------------------------------------------
 * generate_8_rays
 *
 * Genera un Ray8 para los píxeles (x, x+1, ..., x+7) de la fila y.
 * Llama a camera.get_ray() de forma escalar y empaqueta las 8 direcciones
 * en registros AVX2 alineados.
 * Los orígenes son idénticos (posición fija de la cámara) → broadcast.
 * ----------------------------------------------------------------------- */
static Ray8 generate_8_rays(const Camera& cam, int x, int y)
{
    alignas(32) float dx[8], dy[8], dz[8];
    for (int i = 0; i < 8; ++i) {
        Ray r  = cam.get_ray(x + i, y);
        dx[i]  = r.direction.x;
        dy[i]  = r.direction.y;
        dz[i]  = r.direction.z;
    }

    Ray8 r8;
    r8.ox = _mm256_set1_ps(cam.position.x);
    r8.oy = _mm256_set1_ps(cam.position.y);
    r8.oz = _mm256_set1_ps(cam.position.z);
    r8.dx = _mm256_load_ps(dx);
    r8.dy = _mm256_load_ps(dy);
    r8.dz = _mm256_load_ps(dz);
    return r8;
}

/* -----------------------------------------------------------------------
 * write_ppm
 *
 * Escribe framebuffer en PPM binario (P6) con gamma 2.0, idéntico al SEQ.
 * ----------------------------------------------------------------------- */
static void write_ppm(const std::string& fname,
                      const std::vector<Vec3>& fb, int w, int h)
{
    std::ofstream ofs(fname, std::ios::binary);
    if (!ofs) { std::cerr << "[SIMD] Error al escribir " << fname << "\n"; return; }
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        auto to_byte = [](float c) -> unsigned char {
            return static_cast<unsigned char>(
                std::sqrt(std::fmax(0.f, std::fmin(1.f, c))) * 255.f + 0.5f);
        };
        ofs << to_byte(fb[i].x) << to_byte(fb[i].y) << to_byte(fb[i].z);
    }
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    int W = 1600, H = 900;
    std::string out_file = "output_simd_dummy.ppm";
    bool do_validate = false;

    /* Uso: renderer_simd_dummy [width height [output.ppm] [--validate]] */
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--validate") { do_validate = true; }
        else if (i == 1)        { W = std::stoi(arg); }
        else if (i == 2)        { H = std::stoi(arg); }
        else if (i == 3 && arg[0] != '-') { out_file = arg; }
    }

    /* Modo validación: comparar SIMD vs SEQ canal a canal */
    if (do_validate) {
        validate_vs_seq(W, H);
        return 0;
    }

    Scene  scene  = Scene::make_default();
    Camera camera = Camera::make(
        {0.f, 0.5f,  3.f},
        {0.f, 0.f,  -2.f},
        {0.f, 1.f,   0.f},
        50.f, W, H);

    std::vector<Vec3> framebuffer(W * H);

    /* x8: mayor múltiplo de 8 <= W
     * [0, x8)  → vectorizado  |  [x8, W) → scalar fallback */
    const int x8 = (W / 8) * 8;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < H; ++y) {

        /* ---- Parte vectorizada: 8 píxeles por iteración ---- */
        for (int x = 0; x < x8; x += 8) {

            /* LOAD: 8 rayos primarios */
            Ray8 rays = generate_8_rays(camera, x, y);

            /* ISECT + SHADOW + SHADE + REFLECT:
             * pipeline completo → Phong + shadows + reflexión (módulos 2-5) */
            __m256 col_r, col_g, col_b;
            trace_8(rays, scene, col_r, col_g, col_b);

            /* STORE: desempaquetar al framebuffer */
            alignas(32) float r_out[8], g_out[8], b_out[8];
            _mm256_store_ps(r_out, col_r);
            _mm256_store_ps(g_out, col_g);
            _mm256_store_ps(b_out, col_b);
            for (int i = 0; i < 8; ++i)
                framebuffer[y * W + x + i] = {r_out[i], g_out[i], b_out[i]};
        }

        /* ---- Módulo 6: Scalar fallback — Phong completo (idéntico al SEQ) ---- */
        for (int x = x8; x < W; ++x) {
            Ray ray = camera.get_ray(x, y);
            framebuffer[y * W + x] = sc_trace(ray, scene, RT_MAX_DEPTH);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[SIMD] " << W << "x" << H
              << " | AVX2 8px/iter (Phong + shadows + reflexion depth=" << RT_MAX_DEPTH << ")"
              << " | scalar fallback=" << (W - x8) << "px/fila"
              << " | time: " << ms << " ms"
              << " | Mpix/s: " << (W * H / 1e6 / (ms / 1000.0))
              << "\n";

    write_ppm(out_file, framebuffer, W, H);
    std::cout << "[SIMD] Imagen escrita en: " << out_file << "\n";
    return 0;
}
