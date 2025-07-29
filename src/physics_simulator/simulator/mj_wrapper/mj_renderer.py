"""
Useful classes for supporting DeepMind MuJoCo binding.
"""

import os

# DIRTY HACK copied from mujoco-py - a global lock on rendering
from threading import Lock

import mujoco
import numpy as np

_MjSim_render_lock = Lock()

import ctypes
import ctypes.util
import platform

_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"

if os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:
    # If gpu rendering is specified in macros, then we enforce gpu
    # option for rendering
    if _SYSTEM == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    elif _SYSTEM == "Windows":
        os.environ["MUJOCO_GL"] = "wgl"
    else:
        os.environ["MUJOCO_GL"] = "egl"

_MUJOCO_GL = os.environ.get("MUJOCO_GL", "").lower().strip()


class MjRenderContext:
    """
    Class that encapsulates rendering functionality for a
    MuJoCo simulation.

    See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjrendercontext.pyx
    """

    def __init__(
        self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480
    ):

        # move this logic from outside to inside class to avoid multiprocessing issues
        if _MUJOCO_GL not in ("disable", "disabled", "off", "false", "0"):
            _VALID_MUJOCO_GL = ("enable", "enabled", "on", "true", "1", "glfw", "")
            if _SYSTEM == "Linux":
                _VALID_MUJOCO_GL += ("glx", "egl", "osmesa")
            elif _SYSTEM == "Windows":
                _VALID_MUJOCO_GL += ("wgl",)
            elif _SYSTEM == "Darwin":
                _VALID_MUJOCO_GL += ("cgl",)
            if _MUJOCO_GL not in _VALID_MUJOCO_GL:
                raise RuntimeError(
                    f"invalid value for environment variable MUJOCO_GL: {_MUJOCO_GL}"
                )
            # if _SYSTEM == "Linux" and _MUJOCO_GL == "osmesa":
            #     from .context.osmesa_context import OSMesaGLContext as GLContext
            # elif _SYSTEM == "Linux" and _MUJOCO_GL == "egl":
            #     from .context.egl_context import EGLGLContext as GLContext
            # else:
            #     from .context.glfw_context import GLFWGLContext as GLContext

            if _SYSTEM == "Darwin" and _MUJOCO_GL == "cgl":
                from ..mj_context.cgl_context import CGLGLContext as GLContext
            else:
                from ..mj_context.glfw_context import GLFWGLContext as GLContext

        assert offscreen, "only offscreen supported for now"
        self.sim = sim
        self.offscreen = offscreen
        self.device_id = device_id

        # setup GL context with defaults for now
        self.gl_ctx = GLContext(
            max_width=max_width, max_height=max_height, device_id=self.device_id
        )
        self.gl_ctx.make_current()

        # Ensure the model data has been updated so that there
        # is something to render
        sim.forward()
        # make sure sim has this context
        sim.add_render_context(self)

        self.model = sim.model
        self.data = sim.data

        # create default scene
        # set maxgeom to 10k to support large-scale scenes
        self.scn = mujoco.MjvScene(sim.model._model, maxgeom=10000)

        # camera
        self.cam = mujoco.MjvCamera()
        self.cam.fixedcamid = 0
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # options for visual / collision mesh can be set externally, e.g. vopt.geomgroup[0], vopt.geomgroup[1]
        self.vopt = mujoco.MjvOption()

        self.vopt.geomgroup[0] = False  # collision
        self.vopt.geomgroup[1] = True  # visual

        self.pert = mujoco.MjvPerturb()
        self.pert.active = 0
        self.pert.select = 0
        self.pert.skinselect = -1

        # self._markers = []
        # self._overlay = {}

        self._set_mujoco_context_and_buffers()

    def _set_mujoco_context_and_buffers(self):
        self.con = mujoco.MjrContext(
            self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def update_offscreen_size(self, width, height):
        if (width != self.con.offWidth) or (height != self.con.offHeight):
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            self.con.free()
            del self.con
            self._set_mujoco_context_and_buffers()

    def render(self, width, height, camera_id=None, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)

        # if self.sim.render_callback is not None:
        #     self.sim.render_callback(self.sim, self)

        # update width and height of rendering context if necessary
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model._model,
            self.data._data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        # for marker_params in self._markers:
        #     self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)
        # for gridpos, (text1, text2) in self._overlay.items():
        #     mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None

        mujoco.mjr_readPixels(
            rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con
        )

        ret_img = rgb_img
        if segmentation:
            uint32_rgb_img = rgb_img.astype(np.int32)
            seg_img = (
                uint32_rgb_img[:, :, 0]
                + uint32_rgb_img[:, :, 1] * (2**8)
                + uint32_rgb_img[:, :, 2] * (2**16)
            )
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            return (ret_img, depth_img)
        else:
            return ret_img

    def upload_texture(self, tex_id):
        """Uploads given texture to the GPU."""
        self.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.model._model, self.con, tex_id)

    def __del__(self):
        # free mujoco rendering context and GL rendering context
        self.con.free()
        try:
            self.gl_ctx.free()
        except Exception:
            # avoid getting OpenGL.error.GLError
            pass
        del self.con
        del self.gl_ctx
        del self.scn
        del self.cam
        del self.vopt
        del self.pert


class MjRenderContextOffscreen(MjRenderContext):
    def __init__(self, sim, device_id, max_width=640, max_height=480):
        super().__init__(
            sim,
            offscreen=True,
            device_id=device_id,
            max_width=max_width,
            max_height=max_height,
        )
