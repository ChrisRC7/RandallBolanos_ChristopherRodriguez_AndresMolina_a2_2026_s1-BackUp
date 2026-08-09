#pragma once
#include <cmath>

/*
 * Vec3 — vector 3D de flotantes de precisión simple.
 * Tipo base compartido por las implementaciones secuencial, SIMD y CUDA.
 * Compatible con C++17 y CUDA (añadir __host__ __device__ al portar a .cuh).
 */
struct Vec3 {
    float x, y, z;

    Vec3() : x(0.f), y(0.f), z(0.f) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    /* ---- Aritmética elemental ---- */
    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(float t)       const { return {x * t,   y * t,   z * t};   }
    Vec3 operator/(float t)       const { float inv = 1.f / t; return *this * inv; }
    /* Multiplicación componente a componente — útil para color * color */
    Vec3 operator*(const Vec3& v) const { return {x * v.x, y * v.y, z * v.z}; }
    Vec3 operator-()              const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator*=(float t)       { x *= t;   y *= t;   z *= t;   return *this; }

    /* ---- Álgebra lineal ---- */
    float dot(const Vec3& v)  const { return x * v.x + y * v.y + z * v.z; }
    Vec3  cross(const Vec3& v) const {
        return { y * v.z - z * v.y,
                 z * v.x - x * v.z,
                 x * v.y - y * v.x };
    }
    float length_sq() const { return x * x + y * y + z * z; }
    float length()    const { return std::sqrt(length_sq()); }
    Vec3  normalize() const { return *this / length(); }

    /*
     * reflect(v, n): refleja el vector v alrededor de la normal n (unitaria).
     * Convención: v apunta hacia la superficie (v·n < 0).
     * Resultado: vector reflejado que apunta alejándose de la superficie.
     */
    static Vec3 reflect(const Vec3& v, const Vec3& n) {
        return v - n * (2.f * v.dot(n));
    }

    /* Aplica gamma-clamp: cada componente se satura a [0, 1] */
    Vec3 clamp01() const {
        return { std::fmax(0.f, std::fmin(1.f, x)),
                 std::fmax(0.f, std::fmin(1.f, y)),
                 std::fmax(0.f, std::fmin(1.f, z)) };
    }
};

/* Permite escribir: float * Vec3 */
inline Vec3 operator*(float t, const Vec3& v) { return v * t; }
