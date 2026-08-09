/*
 * renderer_cuda_dummy.cu — Kernel CUDA completo para ray tracing.
 * Demo 2 — CE 4302 Arquitectura de Computadores II.
 *
 * Demuestra el esquema DLP en GPU (NVIDIA RTX 4060, sm_89):
 *   — 1 hilo CUDA por píxel, bloques 2D de 16×16 hilos.
 *   — Generación del rayo dentro del kernel (sin transferencias por rayo).
 *   — Intersección rayo-esfera en GPU (misma matemática que versión SEQ).
 *   — Iluminación Phong con sombras y reflexión recursiva.
 *   — Escena en memoria constante (__constant__) para acceso broadcast.
 *   — Framebuffer Structure-of-Arrays (SoA) para maximizar coalescencia.
 *   — Tiempo medido con CUDA Events (solo tiempo del kernel, sin PCIe).
 *
 * Esta versión implementa el mismo modelo de iluminación que la versión SEQ:
 *   — Esferas con material Phong + reflectividad.
 *   — Rayos de sombra por fuente puntual.
 *   — Profundidad máxima de 3 rebotes.
 *
 * Compilar (Linux, CUDA Toolkit 12.x+):
 *   nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
 *        src/cuda/renderer_cuda_dummy.cu -o bin/renderer_cuda_dummy
 *
 * NOTA: Requiere CUDA Toolkit instalado (nvcc). Descarga en:
 *   https://developer.nvidia.com/cuda-downloads
 *   Seleccionar: Linux / x86_64 / Ubuntu / runfile
 *   RTX 4060 = compute capability 8.9 → usar -arch=sm_89
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>   /* atoi */

/* ---- Headers del proyecto (lado host) ---- */
#include "vec3.h"
#include "ray.h"
#include "hit.h"
#include "material.h"
#include "sphere.h"
#include "light.h"
#include "camera.h"
#include "scene.h"


/* ---- Constantes de resolución (defecto) ---- */
static constexpr int DEF_WIDTH  = 1600;
static constexpr int DEF_HEIGHT = 900;
static constexpr int MAX_DEPTH  = 3;
static constexpr float T_MIN    = 1e-4f;
static constexpr float T_MAX    = 1e30f;

/* -----------------------------------------------------------------------
 * Vec3f — Vector 3D de flotantes compatible con __host__ y __device__.
 * Definición independiente de vec3.h para evitar incompatibilidades de
 * cabeceras C++ con el compilador nvcc.
 * ----------------------------------------------------------------------- */
struct Vec3f {
    float x, y, z;

    __host__ __device__ Vec3f() = default;   /* trivially constructible → OK en __constant__ */
    __host__ __device__ Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3f operator+(const Vec3f& v) const { return {x+v.x, y+v.y, z+v.z}; }
    __host__ __device__ Vec3f operator-(const Vec3f& v) const { return {x-v.x, y-v.y, z-v.z}; }
    __host__ __device__ Vec3f operator*(float t)        const { return {x*t,   y*t,   z*t};   }
    __host__ __device__ Vec3f operator*(const Vec3f& v)  const { return {x*v.x, y*v.y, z*v.z}; }
    __host__ __device__ Vec3f operator/(float t)        const { float inv=1.f/t; return *this*inv; }
    __host__ __device__ Vec3f operator-()               const { return {-x, -y, -z}; }

    __host__ __device__ Vec3f& operator+=(const Vec3f& v) { x += v.x; y += v.y; z += v.z; return *this; }

    __host__ __device__ float dot(const Vec3f& v) const { return x*v.x + y*v.y + z*v.z; }
    __host__ __device__ Vec3f cross(const Vec3f& v) const {
        return { y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x };
    }
    __host__ __device__ float length_sq() const { return x*x + y*y + z*z; }
    __host__ __device__ float length()    const { return sqrtf(length_sq()); }
    __host__ __device__ Vec3f normalize() const { return *this / length(); }
    __host__ __device__ Vec3f clamp01() const {
        return {fmaxf(0.f, fminf(1.f, x)),
                fmaxf(0.f, fminf(1.f, y)),
                fmaxf(0.f, fminf(1.f, z))};
    }

    __host__ __device__ static Vec3f reflect(const Vec3f& v, const Vec3f& n) {
        return v - n * (2.f * v.dot(n));
    }
};
__host__ __device__ inline Vec3f operator*(float t, const Vec3f& v) { return v * t; }

struct Ray_d {
    Vec3f origin;
    Vec3f direction;

    __host__ __device__ Ray_d() = default;
    __host__ __device__ Ray_d(const Vec3f& o, const Vec3f& d) : origin(o), direction(d) {}

    __host__ __device__ Vec3f at(float t) const { return origin + direction * t; }
};

struct Material_d {
    Vec3f albedo;
    float Ka;
    float Kd;
    float Ks;
    float shininess;
    float reflectivity;
};

struct Light_d {
    Vec3f position;
    Vec3f intensity;
};

struct HitRecord_d {
    Vec3f point;
    Vec3f normal;
    Material_d material;
    float t;
    bool front_face;
};

/* -----------------------------------------------------------------------
 * Sphere_d — Esfera mínima en dispositivo (solo los datos que usa el kernel).
 * ----------------------------------------------------------------------- */
struct Sphere_d {
    Vec3f center;
    float radius;
    Material_d material;
};

/* -----------------------------------------------------------------------
 * Memoria constante en GPU
 * __constant__ = caché de solo lectura compartida por todos los hilos del SM.
 * Acceso broadcast (todos los hilos del warp leen el mismo dato): ideal para
 * la escena, que es leída por todos los hilos de igual forma.
 * ----------------------------------------------------------------------- */
static constexpr int MAX_SPHERES = 8;
static constexpr int MAX_LIGHTS  = 8;
__constant__ Sphere_d c_spheres[MAX_SPHERES];
__constant__ int      c_num_spheres;
__constant__ Light_d  c_lights[MAX_LIGHTS];
__constant__ int      c_num_lights;

/* Parámetros de cámara precalculados en CPU, almacenados en constante */
__constant__ Vec3f c_cam_pos;   /* posición del ojo                    */
__constant__ Vec3f c_cam_ll;    /* lower_left del viewport             */
__constant__ Vec3f c_cam_h;     /* vector horizontal del viewport      */
__constant__ Vec3f c_cam_v;     /* vector vertical del viewport        */
__constant__ Vec3f c_ambient;   /* luz ambiental global                */
__constant__ Vec3f c_bg;        /* color de fondo (cielo)              */

__device__ __forceinline__ bool intersect_sphere(const Ray_d& ray,
                                                 const Sphere_d& s,
                                                 float t_min,
                                                 float t_max,
                                                 HitRecord_d& rec)
{
    Vec3f oc  = ray.origin - s.center;
    float a   = ray.direction.length_sq();
    float hb  = oc.dot(ray.direction);
    float c   = oc.length_sq() - s.radius * s.radius;
    float disc = hb * hb - a * c;

    if (disc < 0.f) return false;

    float sq = sqrtf(disc);
    float t  = (-hb - sq) / a;
    if (t < t_min || t > t_max) {
        t = (-hb + sq) / a;
        if (t < t_min || t > t_max) return false;
    }

    rec.t = t;
    rec.point = ray.at(t);
    Vec3f outward_normal = (rec.point - s.center) / s.radius;
    rec.front_face = ray.direction.dot(outward_normal) < 0.f;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
    rec.material = s.material;
    return true;
}

__device__ __forceinline__ bool intersect_scene(const Ray_d& ray,
                                                float t_max,
                                                HitRecord_d& rec)
{
    HitRecord_d tmp;
    bool  hit_any = false;
    float closest = t_max;

    for (int i = 0; i < c_num_spheres; ++i) {
        if (intersect_sphere(ray, c_spheres[i], T_MIN, closest, tmp)) {
            hit_any = true;
            closest = tmp.t;
            rec = tmp;
        }
    }
    return hit_any;
}

__device__ __forceinline__ bool in_shadow(const Vec3f& origin,
                                          const Vec3f& light_pos)
{
    Vec3f to_light = light_pos - origin;
    float light_dist = to_light.length();
    Ray_d shadow_ray(origin, to_light / light_dist);

    HitRecord_d dummy;
    for (int i = 0; i < c_num_spheres; ++i) {
        if (intersect_sphere(shadow_ray, c_spheres[i], T_MIN, light_dist - T_MIN, dummy))
            return true;
    }
    return false;
}

__device__ __forceinline__ Vec3f phong_shade(const HitRecord_d& hit,
                                             const Light_d& light,
                                             const Vec3f& ray_dir)
{
    const Material_d& mat = hit.material;
    Vec3f L = (light.position - hit.point).normalize();
    Vec3f R = Vec3f::reflect(-L, hit.normal);
    Vec3f V = -ray_dir;

    float Nd = fmaxf(0.f, hit.normal.dot(L));
    float Ns = powf(fmaxf(0.f, R.dot(V)), mat.shininess);

    Vec3f diffuse  = mat.albedo * (mat.Kd * Nd);
    Vec3f specular = Vec3f(1.f, 1.f, 1.f) * (mat.Ks * Ns);
    return (diffuse + specular) * light.intensity;
}

__device__ __forceinline__ Vec3f trace_ray(const Ray_d& ray, int depth)
{
    if (depth == 0) return {0.f, 0.f, 0.f};

    HitRecord_d rec{};
    if (!intersect_scene(ray, T_MAX, rec))
        return c_bg;

    const Material_d& mat = rec.material;
    Vec3f color = mat.albedo * (c_ambient * mat.Ka);

    for (int i = 0; i < c_num_lights; ++i) {
        if (!in_shadow(rec.point, c_lights[i].position))
            color += phong_shade(rec, c_lights[i], ray.direction);
    }

    if (mat.reflectivity > 0.f) {
        Vec3f reflect_dir = Vec3f::reflect(ray.direction, rec.normal).normalize();
        color += trace_ray(Ray_d(rec.point, reflect_dir), depth - 1) * mat.reflectivity;
    }

    return color.clamp01();
}

/* -----------------------------------------------------------------------
 * ray_trace_kernel
 *
 * Cada hilo procesa exactamente 1 píxel:
 *   1. Calcular índice (px, py) desde blockIdx + threadIdx.
 *   2. Generar dirección del rayo primario.
 *   3. Traza completa: intersección + Phong + sombras + reflexión.
 *   4. Escribir color al framebuffer SoA (fb_r, fb_g, fb_b separados).
 *
 * Diseño SoA (Structure of Arrays):
 *   — fb_r[i], fb_g[i], fb_b[i] en arrays separados.
 *   — Garantiza accesos coalescentes: hilos adyacentes (warp) acceden a
 *     posiciones contiguas de memoria → máximo ancho de banda.
 * ----------------------------------------------------------------------- */
__global__ void ray_trace_kernel(float* __restrict__ fb_r,
                                  float* __restrict__ fb_g,
                                  float* __restrict__ fb_b,
                                  int W, int H)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    /* Guardia de límites (para bloques que exceden la resolución) */
    if (px >= W || py >= H) return;

    /* Coordenadas normalizadas del centro del píxel [0,1] */
    const float u = (px + 0.5f) / (float)W;
    const float v = 1.f - (py + 0.5f) / (float)H;  /* flip Y: py=0 = fila superior */

    /* Dirección del rayo primario (desde cámara al viewport) */
    Vec3f target = c_cam_ll + c_cam_h * u + c_cam_v * v;
    Vec3f dir    = (target - c_cam_pos).normalize();

    Vec3f color = trace_ray(Ray_d(c_cam_pos, dir), MAX_DEPTH);

    /* Escritura al framebuffer SoA */
    const int idx = py * W + px;
    fb_r[idx] = color.x;
    fb_g[idx] = color.y;
    fb_b[idx] = color.z;
}

/* -----------------------------------------------------------------------
 * check_cuda — macro de comprobación de errores CUDA
 * ----------------------------------------------------------------------- */
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "[CUDA] Error en %s:%d — %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while(0)

/* -----------------------------------------------------------------------
 * write_ppm — escribe framebuffer SoA como PPM con corrección gamma 2.0
 * ----------------------------------------------------------------------- */
static void write_ppm(const std::string& fname,
                      const std::vector<float>& r,
                      const std::vector<float>& g,
                      const std::vector<float>& b,
                      int W, int H)
{
    std::ofstream ofs(fname, std::ios::binary);
    if (!ofs) { std::cerr << "[CUDA] Error al escribir " << fname << "\n"; return; }
    ofs << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; ++i) {
        auto to_byte = [](float c) -> unsigned char {
            return static_cast<unsigned char>(
                sqrtf(fmaxf(0.f, fminf(1.f, c))) * 255.f + 0.5f);
        };
        ofs << to_byte(r[i]) << to_byte(g[i]) << to_byte(b[i]);
    }
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    /* Uso: renderer_cuda_dummy [width height [output.ppm]] */
    int WIDTH  = DEF_WIDTH;
    int HEIGHT = DEF_HEIGHT;
    std::string out_file = "output_cuda_dummy.ppm";
    if (argc >= 3) { WIDTH = std::atoi(argv[1]); HEIGHT = std::atoi(argv[2]); }
    if (argc >= 4) out_file = argv[3];

    /* ---- Imprimir info de GPU ---- */
    int device_id = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "[CUDA] GPU: " << prop.name
              << " | SM " << prop.major << "." << prop.minor
              << " | SMs: " << prop.multiProcessorCount
              << " | Mem: " << prop.totalGlobalMem / (1024*1024) << " MB\n";

    /* ---- Construir escena (misma que Scene::make_default) ---- */
    Scene scene = Scene::make_default();
    if (scene.spheres.size() > MAX_SPHERES || scene.lights.size() > MAX_LIGHTS) {
        std::cerr << "[CUDA] Error: la escena excede los limites de memoria constante\n";
        return EXIT_FAILURE;
    }

    std::vector<Sphere_d> h_spheres(scene.spheres.size());
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const Sphere& s = scene.spheres[i];
        h_spheres[i] = {
            {s.center.x, s.center.y, s.center.z},
            s.radius,
            {{s.material.albedo.x, s.material.albedo.y, s.material.albedo.z},
             s.material.Ka, s.material.Kd, s.material.Ks,
             s.material.shininess, s.material.reflectivity}
        };
    }

    std::vector<Light_d> h_lights(scene.lights.size());
    for (size_t i = 0; i < scene.lights.size(); ++i) {
        const PointLight& l = scene.lights[i];
        h_lights[i] = {{l.position.x, l.position.y, l.position.z},
                       {l.intensity.x, l.intensity.y, l.intensity.z}};
    }

    int num_spheres = static_cast<int>(h_spheres.size());
    int num_lights  = static_cast<int>(h_lights.size());

    CHECK_CUDA(cudaMemcpyToSymbol(c_spheres,     h_spheres.data(), h_spheres.size() * sizeof(Sphere_d)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_num_spheres, &num_spheres,     sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_lights,      h_lights.data(),  h_lights.size() * sizeof(Light_d)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_num_lights,  &num_lights,      sizeof(int)));

    /* ---- Calcular cámara (mismos parámetros que Camera::make en camera.h) ---- */
    Vec3f pos      = {0.f, 0.5f,  3.f};
    Vec3f look_at  = {0.f, 0.f,  -2.f};
    Vec3f world_up = {0.f, 1.f,   0.f};

    const float vfov_deg = 50.f;
    const float theta    = vfov_deg * (3.14159265358979f / 180.f);
    const float half_h   = tanf(theta / 2.f);
    const float half_w   = ((float)WIDTH / (float)HEIGHT) * half_h;

    Vec3f w_ax = (pos - look_at).normalize();
    Vec3f u_ax = world_up.cross(w_ax).normalize();
    Vec3f v_ax = w_ax.cross(u_ax);

    Vec3f lower_left = pos - w_ax - u_ax * half_w - v_ax * half_h;
    Vec3f horizontal = u_ax * (2.f * half_w);
    Vec3f vertical   = v_ax * (2.f * half_h);
    Vec3f bg         = {scene.background.x, scene.background.y, scene.background.z};
    Vec3f ambient    = {scene.ambient_light.x, scene.ambient_light.y, scene.ambient_light.z};

    CHECK_CUDA(cudaMemcpyToSymbol(c_cam_pos, &pos,         sizeof(Vec3f)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_cam_ll,  &lower_left,  sizeof(Vec3f)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_cam_h,   &horizontal,  sizeof(Vec3f)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_cam_v,   &vertical,    sizeof(Vec3f)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_ambient, &ambient,     sizeof(Vec3f)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_bg,      &bg,          sizeof(Vec3f)));

    /* ---- Framebuffer en GPU (SoA) ---- */
    const int N = WIDTH * HEIGHT;
    float *d_r, *d_g, *d_b;
    CHECK_CUDA(cudaMalloc(&d_r, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));

    /* ---- Configuración de bloques: 16×16 hilos (256 hilos/bloque) ---- */
    dim3 block(16, 16);
    dim3 grid((WIDTH  + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);

    std::cout << "[CUDA] Grid: " << grid.x << "x" << grid.y
              << " bloques | Block: " << block.x << "x" << block.y
              << " hilos | Total hilos: "
              << grid.x * grid.y * block.x * block.y << "\n";

    /* ---- Lanzar kernel y medir con CUDA Events ---- */
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start));
    ray_trace_kernel<<<grid, block>>>(d_r, d_g, d_b, WIDTH, HEIGHT);
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    CHECK_CUDA(cudaGetLastError());  /* verificar errores del kernel */

    float kernel_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

    std::cout << "[CUDA] " << WIDTH << "x" << HEIGHT
              << " | kernel time: " << kernel_ms << " ms"
              << " | depth=" << MAX_DEPTH
              << " | Phong + shadows + reflection"
              << " (sin transferencias PCIe)\n";

    /* ---- Copiar framebuffer GPU → CPU y medir tiempo PCIe ---- */
    cudaEvent_t ev_h2d_s, ev_h2d_e;
    CHECK_CUDA(cudaEventCreate(&ev_h2d_s));
    CHECK_CUDA(cudaEventCreate(&ev_h2d_e));

    std::vector<float> h_r(N), h_g(N), h_b(N);
    CHECK_CUDA(cudaEventRecord(ev_h2d_s));
    CHECK_CUDA(cudaMemcpy(h_r.data(), d_r, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_g.data(), d_g, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_h2d_e));
    CHECK_CUDA(cudaEventSynchronize(ev_h2d_e));

    float pcie_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&pcie_ms, ev_h2d_s, ev_h2d_e));

    const float total_ms = kernel_ms + pcie_ms;
    std::cout << "[CUDA] PCIe D→H: " << pcie_ms << " ms"
              << " | TOTAL: " << total_ms << " ms"
              << " | Mpix/s (total): " << (WIDTH * HEIGHT / 1e6f / (total_ms / 1000.f))
              << "\n";

    /* ---- Limpiar recursos ---- */
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_g));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaEventDestroy(ev_h2d_s));
    CHECK_CUDA(cudaEventDestroy(ev_h2d_e));

    /* ---- Escribir imagen ---- */
    write_ppm(out_file, h_r, h_g, h_b, WIDTH, HEIGHT);
    std::cout << "[CUDA] Imagen escrita en: " << out_file << "\n";

    return 0;
}
