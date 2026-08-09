# =============================================================================
# Makefile — DLP Ray Tracing Framework
# CE 4302 Arquitectura de Computadores II
#
# Targets:
#   make seq        — compila el renderer secuencial (baseline)
#   make simd       — compila el prototipo SIMD/AVX2 (dummy Demo 2)
#   make cuda       — compila el prototipo CUDA (requiere nvcc)
#   make run_seq    — compila y ejecuta el renderer secuencial
#   make run_simd   — compila y ejecuta el prototipo SIMD
#   make run_cuda   — compila y ejecuta el prototipo CUDA
#   make clean      — elimina binarios e imágenes PPM generadas
#
# Requisitos: GCC/G++ con soporte C++17
# Windows: usar mingw32-make  |  Linux/WSL: usar make
# =============================================================================

CXX       := g++
ifeq ($(OS),Windows_NT)
    NVCC      := "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc"
    NVCC_CCBIN := C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64
else
    NVCC      := nvcc
    NVCC_CCBIN :=
endif

# Detección de SO para comandos de shell compatibles
ifeq ($(OS),Windows_NT)
    MKDIR_P   = if not exist $(BIN_DIR) mkdir $(BIN_DIR)
    RM_ALL    = if exist $(BIN_DIR) rd /s /q $(BIN_DIR)
    RM_PPM    = del /q *.ppm 2>nul || true
else
    MKDIR_P   = mkdir -p $(BIN_DIR)
    RM_ALL    = rm -rf $(BIN_DIR)/
    RM_PPM    = rm -f *.ppm
endif

# Flags base (optimización estándar para la versión secuencial)
CXXFLAGS_SEQ  := -std=c++17 -O2 -Wall -Wextra

# Flags para SIMD/AVX2 — se activan en la iteración SIMD
CXXFLAGS_SIMD := -std=c++17 -O3 -march=native -mavx2 -mfma -Wall -Wextra

# Flags para CUDA — Windows usa sm_89 (RTX 4060); Linux auto-detecta la GPU con -arch=native
ifeq ($(OS),Windows_NT)
    NVCCFLAGS := -O2 -std=c++17 -arch=sm_89 -ccbin "$(NVCC_CCBIN)" -diag-suppress 20012
else
    NVCCFLAGS := -O2 -std=c++17 -arch=sm_50 -diag-suppress 20012 # Linea antigüa: NVCCFLAGS := -O2 -std=c++17 -arch=native -diag-suppress 20012
endif

INC_DIR   := include
BIN_DIR   := bin

HEADERS   := $(wildcard $(INC_DIR)/*.h)

# -----------------------------------------------------------------------------
# Sequential baseline
# -----------------------------------------------------------------------------
SRC_SEQ   := src/sequential/renderer_seq.cpp
BIN_SEQ   := $(BIN_DIR)/renderer_seq

.PHONY: all seq simd cuda run_seq run_simd run_cuda run_all bench results clean

all: seq simd cuda

run_all: run_seq run_simd run_cuda

# -----------------------------------------------------------------------------
# Sequential baseline
# -----------------------------------------------------------------------------
seq: $(BIN_SEQ)

$(BIN_SEQ): $(SRC_SEQ) $(HEADERS)
	@$(MKDIR_P)
	$(CXX) $(CXXFLAGS_SEQ) -I$(INC_DIR) $< -o $@
	@echo "[BUILD] OK -> $@"

run_seq: seq
	$(BIN_SEQ) 1600 900 output_seq.ppm

# -----------------------------------------------------------------------------
# SIMD / AVX2 — prototipo dummy Demo 2
# -----------------------------------------------------------------------------
SRC_SIMD  := src/simd/renderer_simd_dummy.cpp
BIN_SIMD  := $(BIN_DIR)/renderer_simd_dummy

simd: $(BIN_SIMD)

SIMD_HEADERS := $(wildcard src/simd/*.h)

$(BIN_SIMD): $(SRC_SIMD) $(HEADERS) $(SIMD_HEADERS)
	@$(MKDIR_P)
	$(CXX) $(CXXFLAGS_SIMD) -I$(INC_DIR) -Isrc/simd $< -o $@
	@echo "[BUILD] OK -> $@"

run_simd: simd
	$(BIN_SIMD) 1600 900 output_simd_dummy.ppm

# -----------------------------------------------------------------------------
# CUDA — prototipo dummy Demo 2
# Requiere: CUDA Toolkit instalado con nvcc en PATH
#   https://developer.nvidia.com/cuda-downloads
#   RTX 4060 = compute capability 8.9
# -----------------------------------------------------------------------------
SRC_CUDA  := src/cuda/renderer_cuda_dummy.cu
BIN_CUDA  := $(BIN_DIR)/renderer_cuda_dummy

cuda: $(BIN_CUDA)

$(BIN_CUDA): $(SRC_CUDA)
	@$(MKDIR_P)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) $< -o $@
	@echo "[BUILD] OK -> $@"

run_cuda: cuda
	$(BIN_CUDA) 1600 900 output_cuda_dummy.ppm

# -----------------------------------------------------------------------------
# Benchmark — escalabilidad, speedup y eficiencia paralela
# Corre las pruebas, exporta CSVs y genera gráficas automáticamente.
# Requisitos: compilar seq/simd/cuda + tener Python con matplotlib
#   pip install matplotlib
# -----------------------------------------------------------------------------
bench: seq simd cuda
	powershell -ExecutionPolicy Bypass -File benchmark.ps1
	python plot_results.py

# Regenerar solo las gráficas (sin volver a medir)
results:
	python plot_results.py

# -----------------------------------------------------------------------------
# CUDA  (descomentar cuando esté lista la implementación)
# -----------------------------------------------------------------------------
# SRC_CUDA  := src/cuda/renderer_cuda.cu
# BIN_CUDA  := $(BIN_DIR)/renderer_cuda
#
# cuda: $(BIN_CUDA)
#
# $(BIN_CUDA): $(SRC_CUDA) $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/*.cuh)
# 	@mkdir -p $(BIN_DIR)
# 	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) $< -o $@
# 	@echo "[BUILD] OK -> $@"
#
# run_cuda: cuda
# 	./$(BIN_CUDA) output_cuda.ppm

# -----------------------------------------------------------------------------
# Limpieza
# -----------------------------------------------------------------------------
clean:
	@$(RM_ALL)
	@$(RM_PPM)
	@echo "[CLEAN] Binarios e imagenes eliminados."
