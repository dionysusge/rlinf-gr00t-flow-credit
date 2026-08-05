#!/usr/bin/env bash

export PROJECT="/data/Wayne/gzw/rlinf_gr00t_n17"
export BULK="/mnt/models/gzw/rlinf_gr00t_n17"
export VENV="/mnt/models/gzw/rlinf_gr00t_n17/envs/rlinf_n17_libero"

export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source "/mnt/models/gzw/rlinf_gr00t_n17/envs/rlinf_n17_libero/bin/activate"

export MS_ASSET_DIR="/mnt/models/gzw/rlinf_gr00t_n17/assets/.maniskill"
export PHYSX_DIR="/mnt/models/gzw/rlinf_gr00t_n17/assets/.sapien/physx/105.1-physx-5.3.1.patch0"

export HF_HOME="/mnt/models/gzw/rlinf_gr00t_n17/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export UV_CACHE_DIR="/mnt/models/gzw/rlinf_gr00t_n17/cache/uv"
export PIP_CACHE_DIR="/mnt/models/gzw/rlinf_gr00t_n17/cache/pip"
export XDG_CACHE_HOME="/mnt/models/gzw/rlinf_gr00t_n17/cache/xdg"
export TMPDIR="/mnt/models/gzw/rlinf_gr00t_n17/tmp"

# 当前机器 EGL 尚未配置完成，暂时不要强制启用。
unset MUJOCO_GL 2>/dev/null || true
unset PYOPENGL_PLATFORM 2>/dev/null || true

# Local rootless GLVND loader
export RLINF_GL_PREFIX="/mnt/models/gzw/rlinf_gr00t_n17/system_libs/glvnd"
export RLINF_GL_LIB="$RLINF_GL_PREFIX/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$RLINF_GL_LIB:${LD_LIBRARY_PATH:-}"

# Explicitly use the system NVIDIA EGL vendor implementation.
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

