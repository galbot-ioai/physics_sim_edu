#####################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
#
# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
#####################################################################################
#          _____             _   _       _   _
#         / ____|           | | | |     | \ | |
#        | (___  _   _ _ __ | |_| |__   |  \| | _____   ____ _
#         \___ \| | | | '_ \| __| '_ \  | . ` |/ _ \ \ / / _` |
#         ____) | |_| | | | | |_| | | | | |\  | (_) \ V / (_| |
#        |_____/ \__, |_| |_|\__|_| |_| |_| \_|\___/ \_/ \__,_|
#                 __/ |
#                |___/
#
#
#####################################################################################
#
# Description: Mujoco model wrapper for Galbot Education Sim
# Date: 2025-04-02
# Note: This file is a modified version of https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/binding_utils.py
#
#####################################################################################

from tempfile import TemporaryDirectory

import mujoco
import os


class _MjModelMeta(type):
    """
    Metaclass which allows MjModel below to delegate to mujoco.MjModel.

    Taken from dm_control: https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/wrapper/core.py#L244
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjModel):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._model, attr)
                    fset = lambda self, value, attr=attr: setattr(
                        self._model, attr, value
                    )
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjModel(metaclass=_MjModelMeta):
    """Wrapper class for a MuJoCo 'mjModel' instance.
    MjModel encapsulates features of the model that are expected to remain
    constant. It also contains simulation and visualization options which may be
    changed occasionally, although this is done explicitly by the user.
    """

    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self, model_ptr):
        """Creates a new MjModel instance from a mujoco.MjModel."""
        self._model = model_ptr

        # make useful mappings such as _body_name2id and _body_id2name
        self.make_mappings()

    def __del__(self):
        # free mujoco model
        del self._model

    """
    Some methods supported by sim.model in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    def _extract_mj_names(self, name_adr, num_obj, obj_type):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
        """

        ### TODO: fix this to use @name_adr like mujoco-py - more robust than assuming IDs are continuous ###

        # objects don't need to be named in the XML, so name might be None
        id2name = {i: None for i in range(num_obj)}
        name2id = {}
        for i in range(num_obj):
            name = mujoco.mj_id2name(self._model, obj_type, i)
            name2id[name] = i
            id2name[i] = name

        # # objects don't need to be named in the XML, so name might be None
        # id2name = { i: None for i in range(num_obj) }
        # name2id = {}
        # for i in range(num_obj):
        #     name = self.model.names[name_adr[i]]
        #     decoded_name = name.decode()
        #     if decoded_name:
        #         obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        #         assert (0 <= obj_id < num_obj) and (id2name[obj_id] is None)
        #         name2id[decoded_name] = obj_id
        #         id2name[obj_id] = decoded_name

        # sort names by increasing id to keep order deterministic
        return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name

    def make_mappings(self):
        """
        Make some useful internal mappings that mujoco-py supported.
        """
        p = self
        self.body_names, self._body_name2id, self._body_id2name = (
            self._extract_mj_names(p.name_bodyadr, p.nbody, mujoco.mjtObj.mjOBJ_BODY)
        )
        self.joint_names, self._joint_name2id, self._joint_id2name = (
            self._extract_mj_names(p.name_jntadr, p.njnt, mujoco.mjtObj.mjOBJ_JOINT)
        )
        self.geom_names, self._geom_name2id, self._geom_id2name = (
            self._extract_mj_names(p.name_geomadr, p.ngeom, mujoco.mjtObj.mjOBJ_GEOM)
        )
        self.site_names, self._site_name2id, self._site_id2name = (
            self._extract_mj_names(p.name_siteadr, p.nsite, mujoco.mjtObj.mjOBJ_SITE)
        )
        self.light_names, self._light_name2id, self._light_id2name = (
            self._extract_mj_names(p.name_lightadr, p.nlight, mujoco.mjtObj.mjOBJ_LIGHT)
        )
        self.camera_names, self._camera_name2id, self._camera_id2name = (
            self._extract_mj_names(p.name_camadr, p.ncam, mujoco.mjtObj.mjOBJ_CAMERA)
        )
        self.actuator_names, self._actuator_name2id, self._actuator_id2name = (
            self._extract_mj_names(
                p.name_actuatoradr, p.nu, mujoco.mjtObj.mjOBJ_ACTUATOR
            )
        )
        self.sensor_names, self._sensor_name2id, self._sensor_id2name = (
            self._extract_mj_names(
                p.name_sensoradr, p.nsensor, mujoco.mjtObj.mjOBJ_SENSOR
            )
        )
        self.tendon_names, self._tendon_name2id, self._tendon_id2name = (
            self._extract_mj_names(
                p.name_tendonadr, p.ntendon, mujoco.mjtObj.mjOBJ_TENDON
            )
        )
        self.mesh_names, self._mesh_name2id, self._mesh_id2name = (
            self._extract_mj_names(p.name_meshadr, p.nmesh, mujoco.mjtObj.mjOBJ_MESH)
        )

    def body_id2name(self, id):
        """Get body name from mujoco body id."""
        if id not in self._body_id2name:
            raise ValueError("No body with id %d exists." % id)
        return self._body_id2name[id]

    def body_name2id(self, name):
        """Get body id from mujoco body name."""
        if name not in self._body_name2id:
            raise ValueError(
                'No "body" with name %s exists. Available "body" names = %s.'
                % (name, self.body_names)
            )
        return self._body_name2id[name]

    def joint_id2name(self, id):
        """Get joint name from mujoco joint id."""
        if id not in self._joint_id2name:
            raise ValueError("No joint with id %d exists." % id)
        return self._joint_id2name[id]

    def joint_name2id(self, name):
        """Get joint id from joint name."""
        if name not in self._joint_name2id:
            raise ValueError(
                'No "joint" with name %s exists. Available "joint" names = %s.'
                % (name, self.joint_names)
            )
        return self._joint_name2id[name]

    def geom_id2name(self, id):
        """Get geom name from  geom id."""
        if id not in self._geom_id2name:
            raise ValueError("No geom with id %d exists." % id)
        return self._geom_id2name[id]

    def geom_name2id(self, name):
        """Get geom id from  geom name."""
        if name not in self._geom_name2id:
            raise ValueError(
                'No "geom" with name %s exists. Available "geom" names = %s.'
                % (name, self.geom_names)
            )
        return self._geom_name2id[name]

    def site_id2name(self, id):
        """Get site name from site id."""
        if id not in self._site_id2name:
            raise ValueError("No site with id %d exists." % id)
        return self._site_id2name[id]

    def site_name2id(self, name):
        """Get site id from site name."""
        if name not in self._site_name2id:
            raise ValueError(
                'No "site" with name %s exists. Available "site" names = %s.'
                % (name, self.site_names)
            )
        return self._site_name2id[name]

    def light_id2name(self, id):
        """Get light name from light id."""
        if id not in self._light_id2name:
            raise ValueError("No light with id %d exists." % id)
        return self._light_id2name[id]

    def light_name2id(self, name):
        """Get light id from light name."""
        if name not in self._light_name2id:
            raise ValueError(
                'No "light" with name %s exists. Available "light" names = %s.'
                % (name, self.light_names)
            )
        return self._light_name2id[name]

    def camera_id2name(self, id):
        """Get camera name from camera id."""
        if id == -1:
            return "free"
        if id not in self._camera_id2name:
            raise ValueError("No camera with id %d exists." % id)
        return self._camera_id2name[id]

    def camera_name2id(self, name):
        """Get camera id from  camera name."""
        if name == "free":
            return -1
        if name not in self._camera_name2id:
            raise ValueError(
                'No "camera" with name %s exists. Available "camera" names = %s.'
                % (name, self.camera_names)
            )
        return self._camera_name2id[name]

    def actuator_id2name(self, id):
        """Get actuator name from actuator id."""
        if id not in self._actuator_id2name:
            raise ValueError("No actuator with id %d exists." % id)
        return self._actuator_id2name[id]

    def actuator_name2id(self, name):
        """Get actuator id from actuator name."""
        if name not in self._actuator_name2id:
            raise ValueError(
                'No "actuator" with name %s exists. Available "actuator" names = %s.'
                % (name, self.actuator_names)
            )
        return self._actuator_name2id[name]

    def sensor_id2name(self, id):
        """Get sensor name from sensor id."""
        if id not in self._sensor_id2name:
            raise ValueError("No sensor with id %d exists." % id)
        return self._sensor_id2name[id]

    def sensor_name2id(self, name):
        """Get sensor id from sensor name."""
        if name not in self._sensor_name2id:
            raise ValueError(
                'No "sensor" with name %s exists. Available "sensor" names = %s.'
                % (name, self.sensor_names)
            )
        return self._sensor_name2id[name]

    def tendon_id2name(self, id):
        """Get tendon name from tendon id."""
        if id not in self._tendon_id2name:
            raise ValueError("No tendon with id %d exists." % id)
        return self._tendon_id2name[id]

    def tendon_name2id(self, name):
        """Get tendon id from tendon name."""
        if name not in self._tendon_name2id:
            raise ValueError(
                'No "tendon" with name %s exists. Available "tendon" names = %s.'
                % (name, self.tendon_names)
            )
        return self._tendon_name2id[name]

    def mesh_id2name(self, id):
        """Get mesh name from  mesh id."""
        if id not in self._mesh_id2name:
            raise ValueError("No mesh with id %d exists." % id)
        return self._mesh_id2name[id]

    def mesh_name2id(self, name):
        """Get mesh id from mesh name."""
        if name not in self._mesh_name2id:
            raise ValueError(
                'No "mesh" with name %s exists. Available "mesh" names = %s.'
                % (name, self.mesh_names)
            )
        return self._mesh_name2id[name]

    def userdata_id2name(self, id):
        if id not in self._userdata_id2name:
            raise ValueError("No userdata with id %d exists." % id)
        return self._userdata_id2name[id]

    def userdata_name2id(self, name):
        if name not in self._userdata_name2id:
            raise ValueError("No \"userdata\" with name %s exists. Available \"userdata\" names = %s." % (name, self.userdata_names))
        return self._userdata_name2id[name]

    def get_xml(self):
        with TemporaryDirectory() as td:
            filename = os.path.join(td, "model.xml")
            ret = mujoco.mj_saveLastXML(filename.encode(), self._model)
            return open(filename).read()

    def get_joint_qpos_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1178

        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        """
        # joint_id = self.joint_name2id(name)
        # WARNING@Herman: changed to actuator_name2id
        joint_id = self.actuator_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            )
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1202

        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        """
        # joint_id = self.joint_name2id(name)
        # WARNING@Herman: changed to actuator_name2id
        joint_id = self.actuator_name2id(name)

        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_dofadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            )
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)
