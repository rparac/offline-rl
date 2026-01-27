#!/bin/bash
# Wrapper script to run alfworld with proper OpenGL/Mesa configuration
# This helps resolve the missing swrast_dri.so driver issue
#
# IMPORTANT: X2Go uses indirect GLX which only supports OpenGL 1.2.
# Unity requires OpenGL 3.2+ core profile, which indirect GLX cannot provide.
# This is a fundamental X2Go limitation - you may need direct rendering or
# an alternative remote desktop solution that supports modern OpenGL.

# Set display (already set, but ensure it's correct)
export DISPLAY=${DISPLAY:-:50}

# Force software rendering - use llvmpipe which supports OpenGL 3.3+ core profile
# llvmpipe provides direct rendering with full OpenGL 3.3 support (vs indirect which only has 1.2)
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Point Mesa to llvmpipe (supports OpenGL 3.3+ core profile)
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

# Disable Vulkan completely - Unity log shows "Unsupported windowing backend 0" with Vulkan
export VK_ICD_FILENAMES=""
export __VK_LAYER_NV_optimus=NVIDIA_only
# Unity-specific Vulkan disable
export UNITY_DISABLE_VULKAN=1
export DISABLE_VULKAN=1

# Try to force Unity to use compatibility profile if core profile not available
# Note: X2Go indirect GLX only supports OpenGL 1.2, so this may not work
export UNITY_OPENGL_FORCE_COMPATIBILITY_PROFILE=1

# Fix library conflict: Mesa drivers need system libraries, not conda ones
# Save original LD_LIBRARY_PATH and filter out conda paths that cause conflicts
ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# Remove conda paths from LD_LIBRARY_PATH to prevent libffi/libLLVM conflicts
CLEAN_LD_LIBRARY_PATH=$(echo "$ORIG_LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "miniconda3\|conda" | tr '\n' ':' | sed 's/:$//')
# Prioritize system library paths for Mesa/OpenGL
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/dri:/lib/x86_64-linux-gnu:${CLEAN_LD_LIBRARY_PATH}"

# Force system libffi to be used first (prevents Mesa from loading conda's incompatible libffi)
# This ensures libLLVM-12.so.1 loads system libffi instead of conda's
if [ -f "/usr/lib/x86_64-linux-gnu/libffi.so.7" ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7:${LD_PRELOAD}"
fi

# Fix XDG runtime directory if needed (can cause window creation issues)
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR=/tmp/runtime-$USER
    mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null
    chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null
fi

# Ensure X11 is accessible
export XAUTHORITY=${XAUTHORITY:-~/.Xauthority}

# Disable GPU acceleration fallback that might cause issues
export __GLX_VENDOR_LIBRARY_NAME=mesa

# Run alfworld-play-thor with all arguments passed through
alfworld-play-thor "$@"
