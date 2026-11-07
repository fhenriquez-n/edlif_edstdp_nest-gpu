#!/usr/bin/env bash
#
# build.sh - compiles and installs the NEST GPU tree in <work>/src.
#
#   <work>/src      NEST GPU source (prepared by the other scripts)
#   <work>/build    build folder
#   <work>/install  installation: source <work>/install/bin/nestgpu_vars.sh to use it
#
# <work> is $NESTGPU_WORK, or <package>/work.
# Needs nvcc: run it where the CUDA toolkit is 
#
# Variables:
#   GPU_ARCH   CUDA architecture to compile for. 80 = A100 (default)
#   JOBS       number of parallel compilation jobs (default 8)
#
# Usage:  bash build.sh

set -euo pipefail

PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${NESTGPU_WORK:-$PACKAGE/work}
GPU_ARCH=${GPU_ARCH:-80}
JOBS=${JOBS:-8}
SRC=$WORK/src

command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 1; }
[ -f "$SRC/CMakeLists.txt" ] || { echo "No NEST GPU in $SRC: run get_nestgpu.sh first" >&2; exit 1; }
[ -f "$SRC/src/edstdp.cu" ] || { echo "ED-STDP is not in the tree: run apply_edstdp.py first" >&2; exit 1; }
ls "$SRC"/src/*_nestml.cu >/dev/null 2>&1 || { echo "No generated models: run generate_gpu.py first" >&2; exit 1; }
if grep -l '0\.001 \* y\[N_SCAL_VAR' "$SRC"/src/*_nestml.cu 2>/dev/null; then
    echo "The generated models above are not patched: run patch_generated.py first" >&2
    exit 1
fi

mkdir -p "$WORK/build"
cd "$WORK/build"
cmake -DCMAKE_INSTALL_PREFIX="$WORK/install" -Dwith-gpu-arch="$GPU_ARCH" ../src
make -j "$JOBS"
make install

echo "BUILD OK (GPU_ARCH=$GPU_ARCH)"
echo "To use it:  source $WORK/install/bin/nestgpu_vars.sh"
