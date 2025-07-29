# Modifications Copyright 2025 The robosuite Authors
# Original Copyright 2018 The dm_control Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import atexit
import ctypes
import os

PYOPENGL_PLATFORM = os.environ.get("PYOPENGL_PLATFORM")

if not PYOPENGL_PLATFORM:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
elif PYOPENGL_PLATFORM.lower() != "egl":
    raise ImportError(
        "Cannot use EGL rendering platform. "
        "The PYOPENGL_PLATFORM environment variable is set to {!r} "
        "(should be either unset or 'egl')."
    )

from mujoco.egl import egl_ext as EGL


def create_initialized_egl_device_display(device_id=0):
    """Creates an initialized EGL display directly on a device."""
    display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if display == EGL.EGL_NO_DISPLAY:
        raise RuntimeError("Failed to get EGL display")
    
    major = ctypes.c_int()
    minor = ctypes.c_int()
    if not EGL.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
        raise RuntimeError("Failed to initialize EGL")
    
    return display


global EGL_DISPLAY
EGL_DISPLAY = None

EGL_ATTRIBUTES = (
    EGL.EGL_RED_SIZE,
    8,
    EGL.EGL_GREEN_SIZE,
    8,
    EGL.EGL_BLUE_SIZE,
    8,
    EGL.EGL_ALPHA_SIZE,
    8,
    EGL.EGL_DEPTH_SIZE,
    24,
    EGL.EGL_STENCIL_SIZE,
    8,
    EGL.EGL_COLOR_BUFFER_TYPE,
    EGL.EGL_RGB_BUFFER,
    EGL.EGL_SURFACE_TYPE,
    EGL.EGL_PBUFFER_BIT,
    EGL.EGL_RENDERABLE_TYPE,
    EGL.EGL_OPENGL_BIT,
    EGL.EGL_NONE,
)


class EGLGLContext:
    """An EGL context for headless accelerated OpenGL rendering on GPU devices."""

    def __init__(self, max_width, max_height, device_id=0):
        del max_width, max_height  # unused
        num_configs = ctypes.c_long()
        config_size = 1
        config_ptr = EGL.EGLConfig()  # Makes an opaque pointer
        EGL.eglReleaseThread()
        global EGL_DISPLAY
        if EGL_DISPLAY is None:
            # only initialize for the first time
            EGL_DISPLAY = create_initialized_egl_device_display(device_id=device_id)
            if EGL_DISPLAY == EGL.EGL_NO_DISPLAY:
                raise ImportError(
                    "Cannot initialize a EGL device display. This likely means that your EGL "
                    "driver does not support the PLATFORM_DEVICE extension, which is "
                    "required for creating a headless rendering context."
                )
            atexit.register(EGL.eglTerminate, EGL_DISPLAY)
        EGL.eglChooseConfig(EGL_DISPLAY, EGL_ATTRIBUTES, config_ptr, config_size, num_configs)
        if num_configs.value < 1:
            raise RuntimeError(
                "EGL failed to find a framebuffer configuration that matches the "
                "desired attributes: {}".format(EGL_ATTRIBUTES)
            )
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        self._context = EGL.eglCreateContext(EGL_DISPLAY, config_ptr, EGL.EGL_NO_CONTEXT, None)
        if not self._context:
            raise RuntimeError("Cannot create an EGL context.")

    def make_current(self):
        if not EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, self._context):
            raise RuntimeError("Failed to make the EGL context current.")

    def free(self):
        """Frees resources associated with this context."""
        if self._context:
            current_context = EGL.eglGetCurrentContext()
            if current_context and self._context.address == current_context.address:
                EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
            EGL.eglDestroyContext(EGL_DISPLAY, self._context)
            EGL.eglReleaseThread()
        self._context = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            # avoid getting OpenGL.error.GLError
            pass
