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
# Description: Mujoco data wrapper for Galbot Education Sim
# Date: 2025-04-02
# Note: This file is a modified version of https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/binding_utils.py
#
#####################################################################################


import mujoco
import numpy as np


class _MjDataMeta(type):
    """
    Metaclass which allows MjData below to delegate to mujoco.MjData.

    Taken from dm_control.
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjData):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._data, attr)
                    fset = lambda self, value, attr=attr: setattr(
                        self._data, attr, value
                    )
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjData(metaclass=_MjDataMeta):
    """Wrapper class for a MuJoCo 'mjData' instance.
    MjData contains all of the dynamic variables and intermediate results produced
    by the simulation. These are expected to change on each simulation timestep.
    The properties without docstrings are defined in mujoco source code from https://github.com/deepmind/mujoco/blob/062cb53a4a14b2a7a900453613a7ce498728f9d8/include/mujoco/mjdata.h#L126.
    """

    def __init__(self, model):
        """Construct a new MjData instance.
        Args:
          model: An MjModel instance.
        """
        self._model = model
        self._data = mujoco.MjData(model._model)

    @property
    def model(self):
        """The parent MjModel for this MjData instance."""
        return self._model

    def __del__(self):
        # free mujoco data
        del self._data

    """
    Some methods supported by sim.data in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    @property
    def body_xpos(self):
        """
        Note: mujoco-py used to support sim.data.body_xpos but DM mujoco bindings requires sim.data.xpos,
              so we explicitly expose this as a property
        """
        return self._data.xpos

    @property
    def body_xquat(self):
        """
        Note: mujoco-py used to support sim.data.body_xquat but DM mujoco bindings requires sim.data.xquat,
              so we explicitly expose this as a property
        """
        return self._data.xquat

    @property
    def body_xmat(self):
        """
        Note: mujoco-py used to support sim.data.body_xmat but DM mujoco bindings requires sim.data.xmax,
              so we explicitly expose this as a property
        """
        return self._data.xmat

    def get_body_xpos(self, name):
        """
        Query cartesian position of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xpos (np.ndarray): The xpos value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xpos[bid]

    def get_body_xquat(self, name):
        """
        Query the rotation of a mujoco body in quaternion (in wxyz convention) using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xquat (np.ndarray): The xquat value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xquat[bid]

    def get_body_xmat(self, name):
        """
        Query the rotation of a mujoco body in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xmat (np.ndarray): The xmat value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xmat[bid].reshape((3, 3))

    def get_body_jacp(self, name):
        """
        Query the position jacobian of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, None, jacr, bid)
        return jacr

    def get_body_xvelp(self, name):
        """
        Query the translational velocity of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco body.
        """
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_body_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco body.
        """
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_geom_xpos(self, name):
        """
        Query the cartesian position of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            geom_xpos (np.ndarray): The cartesian position of the mujoco body.
        """
        gid = self.model.geom_name2id(name)
        return self.geom_xpos[gid]

    def get_geom_xmat(self, name):
        """
        Query the rotation of a mujoco geom in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            geom_xmat (np.ndarray): The 3x3 rotation matrix of the mujoco geom.
        """
        gid = self.model.geom_name2id(name)
        return self.geom_xmat[gid].reshape((3, 3))

    def get_geom_jacp(self, name):
        """
        Query the position jacobian of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco geom
        """
        gid = self.model.geom_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, jacp, None, gid)
        return jacp

    def get_geom_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco geom
        """
        gid = self.model.geom_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, None, jacr, gid)
        return jacr

    def get_geom_xvelp(self, name):
        """
        Query the translational velocity of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco geom
        """
        jacp = self.get_geom_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_geom_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco geom
        """
        jacr = self.get_geom_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_site_xpos(self, name):
        """
        Query the cartesian position of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            site_xpos (np.ndarray): The carteisan position of the mujoco site
        """
        sid = self.model.site_name2id(name)
        return self.site_xpos[sid]

    def get_site_xmat(self, name):
        """
        Query the rotation of a mujoco site in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            site_xmat (np.ndarray): The 3x3 rotation matrix of the mujoco site.
        """
        sid = self.model.site_name2id(name)
        return self.site_xmat[sid].reshape((3, 3))

    def get_site_jacp(self, name):
        """
        Query the position jacobian of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco site
        """
        sid = self.model.site_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco site
        """
        sid = self.model.site_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, None, jacr, sid)
        return jacr

    def get_site_xvelp(self, name):
        """
        Query the translational velocity of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco site
        """
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_site_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco site
        """
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_camera_xpos(self, name):
        """
        Get the cartesian position of a camera using name

        Args:
            name (str): The name of a camera
        Returns:
            cam_xpos (np.ndarray): The cartesian position of a camera
        """
        cid = self.model.camera_name2id(name)
        return self.cam_xpos[cid]

    def get_camera_xmat(self, name):
        """
        Get the rotation of a camera in a rotation matrix using name

        Args:
            name (str): The name of a camera
        Returns:
            cam_xmat (np.ndarray): The 3x3 rotation matrix of a camera
        """
        cid = self.model.camera_name2id(name)
        return self.cam_xmat[cid].reshape((3, 3))

    def get_light_xpos(self, name):
        """
        Get cartesian position of a light source

        Args:
            name (str): The name of a lighting source
        Returns:
            light_xpos (np.ndarray): The cartesian position of the light source
        """
        lid = self.model.light_name2id(name)
        return self.light_xpos[lid]

    def get_light_xdir(self, name):
        """
        Get the direction of a light source using name

        Args:
            name (str): The name of a light
        Returns:
            light_xdir (np.ndarray): The direction vector of the lightsource
        """
        lid = self.model.light_name2id(name)
        return self.light_xdir[lid]

    def get_sensor(self, name):
        """
        Get the data of a sensor using name

        Args:
            name (str): The name of a sensor
        Returns:
            sensordata (np.ndarray): The sensor data vector
        """
        sid = self.model.sensor_name2id(name)
        return self.sensordata[sid]

    def get_mocap_pos(self, name):
        """
        Get the position of a mocap body using name.

        Args:
            name (str): The name of a joint
        Returns:
            mocap_pos (np.ndarray): The current position of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        """
        Set the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint position of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        """
        Get the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
        Returns:
            mocap_quat (np.ndarray): The current quaternion of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        """
        Set the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint quaternion of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_quat[mocap_id] = value

    def get_joint_qpos(self, name):
        """
        Get the position of a joint using name.

        Args:
            name (str): The name of a joint

        Returns:
            qpos (np.ndarray): The current position of a joint.
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qpos[addr]
        else:
            start_i, end_i = addr
            return self.qpos[start_i:end_i]

    def set_joint_qpos(self, name, value):
        """
        Set the position of a joint using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint velocity of a joint.
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qpos[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (
                end_i - start_i,
            ), "Value has incorrect shape %s: %s" % (name, value)
            self.qpos[start_i:end_i] = value

    def get_joint_qvel(self, name):
        """
        Get the velocity of a joint using name.

        Args:
            name (str): The name of a joint

        Returns:
            qvel (np.ndarray): The current velocity of a joint.
        """
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qvel[addr]
        else:
            start_i, end_i = addr
            return self.qvel[start_i:end_i]

    def set_joint_qvel(self, name, value):
        """
        Set the velocities of a joint using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint velocity of a joint.
        """
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qvel[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (
                end_i - start_i,
            ), "Value has incorrect shape %s: %s" % (name, value)
            self.qvel[start_i:end_i] = value
