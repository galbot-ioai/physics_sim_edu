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
#####################################################################################
#
# Description: Object classes for SynthNova Physics Simulator
# Author: Chenyu Cao@Galbot
# Date: 2025-04-01
# 
# Note: This code is adapted from robosuite (https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/objects/objects.py)
# under the MIT License. Original copyright:
# Copyright (c) 2018-2022 ARISE Initiative
#####################################################################################


import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from auro_utils import Logger

from physics_simulator.object import MujocoModel, MujocoXML
from physics_simulator.utils.mjcf_utils import (
    OBJECT_COLLISION_COLOR,
    # CustomMaterial,
    # add_material,
    add_prefix,
    array_to_string,
    find_elements,
    new_joint,
    sort_elements,
    string_to_array,
    CustomMaterial,
    new_body,
    new_geom,
    new_site
)


# Dict mapping geom type string keywords to group number
GEOMTYPE2GROUP = {
    "collision": {0},  # If we want to use a geom for physics, but NOT visualize
    "visual": {1},  # If we want to use a geom for visualization, but NOT physics
    "all": {0, 1},  # If we want to use a geom for BOTH physics + visualization
    # "visual_2": {2},  # visual for mesh
    # "collision_2": {3},  # collision for mesh
}

GEOM_GROUPS = GEOMTYPE2GROUP.keys()


class MujocoObject(MujocoModel):
    """
    Base class for all objects.

    We use Mujoco Objects to implement all objects that:

        1) may appear for multiple times in a task
        2) can be swapped between different tasks

    Typical methods return copy so the caller can all joints/attributes as wanted

    Args:
        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy

    """

    def __init__(self, obj_type="all", duplicate_collision_geoms=True):
        super().__init__()
        self.asset = ET.Element("asset")
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

        # Attributes that should be filled in within the subclass
        self._name = None
        self._obj = None

        # Attributes that are auto-filled by _get_object_properties call
        self._root_body = None
        self._bodies = None
        self._joints = None
        self._actuators = None
        self._sites = None
        self._contact_geoms = None
        self._visual_geoms = None

    def merge_assets(self, other):
        """
        Merges @other's assets in a custom logic.

        Args:
            other (MujocoXML or MujocoObject): other xml file whose assets will be merged into this one
        """
        for asset in other.asset:
            if (
                find_elements(root=self.asset, tags=asset.tag, attribs={"name": asset.get("name")}, return_first=True)
                is None
            ):
                self.asset.append(asset)

    def get_obj(self):
        """
        Returns the generated / extracted object, in XML ElementTree form.

        Returns:
            ET.Element: Object in XML form.
        """
        assert self._obj is not None, "Object XML tree has not been generated yet!"
        return self._obj

    def exclude_from_prefixing(self, inp):
        """
        A function that should take in either an ET.Element or its attribute (str) and return either True or False,
        determining whether the corresponding name / str to @inp should have naming_prefix added to it.
        Must be defined by subclass.

        Args:
            inp (ET.Element or str): Element or its attribute to check for prefixing.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        raise NotImplementedError

    def _get_object_subtree(self):

        """
        Returns a ET.Element
        It is a <body/> subtree that defines all collision and / or visualization related fields
        of this object.
        Return should be a copy.
        Must be defined by subclass.

        Returns:
            ET.Element: body
        """
        raise NotImplementedError

    def _get_object_properties(self):
        """
        Helper function to extract relevant object properties (bodies, joints, contact/visual geoms, etc...) from this
        object's XML tree. Assumes the self._obj attribute has already been filled.
        """
        # Parse element tree to get all relevant bodies, joints, actuators, and geom groups
        _elements = sort_elements(root=self.get_obj())
        assert (
            len(_elements["root_body"]) == 1
        ), "Invalid number of root bodies found for robot model. Expected 1," "got {}".format(
            len(_elements["root_body"])
        )
        _elements["root_body"] = _elements["root_body"][0]
        _elements["bodies"] = (
            [_elements["root_body"]] + _elements["bodies"] if "bodies" in _elements else [_elements["root_body"]]
        )
        self._root_body = _elements["root_body"].get("name")
        self._bodies = [e.get("name") for e in _elements.get("bodies", [])]
        self._joints = [e.get("name") for e in _elements.get("joints", [])]
        self._actuators = [e.get("name") for e in _elements.get("actuators", [])]
        self._sites = [e.get("name") for e in _elements.get("sites", [])]
        self._sensors = [e.get("name") for e in _elements.get("sensors", [])]
        self._contact_geoms = [e.get("name") for e in _elements.get("contact_geoms", [])]
        self._visual_geoms = [e.get("name") for e in _elements.get("visual_geoms", [])]


        # Add prefix to all elements
        add_prefix(root=self.get_obj(), prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    @property
    def name(self):
        return self._name

    @property
    def naming_prefix(self):
        return "{}_".format(self.name)

    @property
    def root_body(self):
        return self.correct_naming(self._root_body)

    @property
    def bodies(self):
        return self.correct_naming(self._bodies)

    @property
    def joints(self):
        return self.correct_naming(self._joints)

    @property
    def actuators(self):
        return self.correct_naming(self._actuators)

    @property
    def sites(self):
        return self.correct_naming(self._sites)

    @property
    def sensors(self):
        return self.correct_naming(self._sensors)

    @property
    def contact_geoms(self):
        return self.correct_naming(self._contact_geoms)

    @property
    def visual_geoms(self):
        return self.correct_naming(self._visual_geoms)

    @property
    def important_geoms(self):
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}

    @property
    def important_sites(self):
        """
        Returns:
            dict:

                :`obj`: Object default site
        """
        return {"obj": self.naming_prefix + "default_site"}

    @property
    def important_sensors(self):
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

    @staticmethod
    def get_site_attrib_template():
        """
        Returns attribs of spherical site used to mark body origin

        Returns:
            dict: Dictionary of default site attributes
        """
        return {
            "pos": "0 0 0",
            "size": "0.002 0.002 0.002",
            "rgba": "1 0 0 1",
            "type": "sphere",
            "group": "0",
        }

    @staticmethod
    def get_joint_attrib_template():
        """
        Returns attribs of free joint

        Returns:
            dict: Dictionary of default joint attributes
        """
        return {
            "type": "free",
        }

    def get_bounding_box_half_size(self):
        raise NotImplementedError

    def get_bounding_box_size(self):
        """
        Returns numpy array with dimensions of a bounding box around this object.
        """
        return 2.0 * self.get_bounding_box_half_size()

    def get_position(self):
        """Get the position of the object in world frame.
        
        Note: This returns a default position. For runtime position queries,
        use simulator.get_object_state() instead.
        
        Returns:
            np.ndarray: 3D position [x, y, z] - default position only
        """
        # Return default position - actual position should be queried through simulator
        # This method is kept for API compatibility but should not be used for runtime queries
        
        # Create a temporary logger for deprecation warning
        logger = Logger()
        logger.log_warning(
            "get_position() called on object - this returns default position only. "
            "Use simulator.get_object_state() for runtime position."
        )
        return np.array([0.0, 0.0, 0.0])
        
    def get_orientation(self):
        """Get the orientation of the object in world frame.
        
        Note: This returns a default orientation. For runtime orientation queries,
        use simulator.get_object_state() instead.
        
        Returns:
            np.ndarray: Quaternion in xyzw format - default orientation only
        """
        # Return default orientation - actual orientation should be queried through simulator
        # This method is kept for API compatibility but should not be used for runtime queries
        
        # Create a temporary logger for deprecation warning
        logger = Logger()
        logger.log_warning(
            "get_orientation() called on object - this returns default orientation only. "
            "Use simulator.get_object_state() for runtime orientation."
        )
        return np.array([0.0, 0.0, 0.0, 1.0])

    def get_translation(self):
        """Get the translation of the object in local frame.
        For simplicity, this returns the same as get_position().
        
        Returns:
            np.ndarray: 3D position [x, y, z]
        """
        return self.get_position()
        
    def get_rotation(self):
        """Get the rotation of the object in local frame.
        For simplicity, this returns the same as get_orientation().
        
        Returns:
            np.ndarray: Quaternion in xyzw format
        """
        return self.get_orientation()

    def set_pos(self, pos):
        """
        Set position of object position is defined as center of bounding box

        Args:
            pos (list of floats): 3D position to set object (should be 3 dims)
        """
        self._obj.set("pos", array_to_string(pos))

    def set_euler(self, euler):
        """
        Set Euler value object position

        Args:
            euler (list of floats): 3D Euler values (should be 3 dims)
        """
        self._obj.set("euler", array_to_string(euler))

    def set_quat(self, quat):
        """
        Set quaternion orientation for object.
        
        Args:
            quat (list of floats): Quaternion [w, x, y, z] in wxyz format (4 dims)
        """
        self._obj.set("quat", array_to_string(quat))


class MujocoXMLObject(MujocoObject, MujocoXML):
    """
    MujocoObjects that are loaded from xml files (by default, inherit all properties (e.g.: name)
    from MujocoObject class first!)

    Args:
        fname (str): XML File path

        name (str): Name of this MujocoXMLObject

        joints (None or str or list of dict): each dictionary corresponds to a joint that will be created for this
            object. The dictionary should specify the joint attributes (type, pos, etc.) according to the MuJoCo xml
            specification. If "default", a single free-joint will be automatically generated. If None, no joints will
            be created.

        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy

        scale (float or list of floats): 3D scale factor
    """

    def __init__(self, fname, name, joints="default", obj_type="all", duplicate_collision_geoms=True, scale=None):
        MujocoXML.__init__(self, fname)
        # Set obj type and duplicate args
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

        # Set name
        self._name = name

        # set scale
        self._scale = scale

        # joints for this object
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints have names!
        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(i)

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # scale
        if self._scale is not None:
            self.set_scale(self._scale)

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def _get_object_subtree(self):
        # Parse object
        # this line used to be wrapped in deepcopy.
        # removed this deepcopy line, as it creates discrepancies between obj and self.worldbody!
        obj = self.worldbody.find("./body/body[@name='object']")
        # Rename this top level object body (will have self.naming_prefix added later)
        obj.attrib["name"] = "main"
        # Get all geom_pairs in this tree
        geom_pairs = self._get_geoms(obj)

        # Define a temp function so we don't duplicate so much code
        obj_type = self.obj_type

        def _should_keep(el):
            return int(el.get("group")) in GEOMTYPE2GROUP[obj_type]

        # Loop through each of these pairs and modify them according to @elements arg
        for i, (parent, element) in enumerate(geom_pairs):
            # Delete non-relevant geoms and rename remaining ones
            if not _should_keep(element):
                parent.remove(element)
            else:
                g_name = element.get("name")
                g_name = g_name if g_name is not None else f"g{i}"
                element.set("name", g_name)
                # Also optionally duplicate collision geoms if requested (and this is a collision geom)
                if self.duplicate_collision_geoms and element.get("group") in {None, "0"}:
                    # parent.append(self._duplicate_visual_from_collision(element))
                    # Also manually set the visual appearances to the original collision model
                    element.set("rgba", array_to_string(OBJECT_COLLISION_COLOR))
                    if element.get("material") is not None:
                        del element.attrib["material"]
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # Lastly, add a site for this object
        template = self.get_site_attrib_template()
        template["rgba"] = "1 0 0 0"
        template["name"] = "default_site"
        obj.append(ET.Element("site", attrib=template))

        return obj

    def exclude_from_prefixing(self, inp):
        """
        By default, don't exclude any from being prefixed
        """
        return False

    def _get_object_properties(self):
        """
        Extends the base class method to also add prefixes to all bodies in this object
        """
        super()._get_object_properties()
        add_prefix(root=self.root, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    @staticmethod
    def _duplicate_visual_from_collision(element):
        """
        Helper function to duplicate a geom element to be a visual element. Namely, this corresponds to the
        following attribute requirements: group=1, conaffinity/contype=0, no mass, name appended with "_visual"

        Args:
            element (ET.Element): element to duplicate as a visual geom

        Returns:
            element (ET.Element): duplicated element
        """
        # Copy element
        vis_element = deepcopy(element)
        # Modify for visual-specific attributes (group=1, conaffinity/contype=0, no mass, update name)
        vis_element.set("group", "1")
        vis_element.set("conaffinity", "0")
        vis_element.set("contype", "0")
        vis_element.set("mass", "1e-8")
        vis_element.set("name", vis_element.get("name") + "_visual")
        return vis_element

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        return self._get_elements(root, "geom", _parent)

    def _get_elements(self, root, type, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a specific type of element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is of type
        """
        # Initialize return array
        elem_pairs = []
        # If the parent exists and this is a desired element, we add this current (parent, element) combo to the output
        if _parent is not None and root.tag == type:
            elem_pairs.append((_parent, root))
        # Loop through all children elements recursively and add to pairs
        for child in root:
            elem_pairs += self._get_elements(child, type, _parent=root)

        # Return all found pairs
        return elem_pairs

    @property
    def rot(self):
        rot = string_to_array(self._obj.get("euler", "0.0 0.0 0.0"))
        return rot[2]

    def set_scale(self, scale, obj=None):
        """
        Scales each geom, mesh, site, and body.
        Called during initialization but can also be used externally

        Args:
            scale (float or list of floats): Scale factor (1 or 3 dims)
            obj (ET.Element) Root object to apply. Defaults to root object of model
        """
        if obj is None:
            obj = self._obj

        self._scale = scale

        # scale geoms
        geom_pairs = self._get_geoms(obj)
        for _, (_, element) in enumerate(geom_pairs):
            g_pos = element.get("pos")
            g_size = element.get("size")
            if g_pos is not None:
                g_pos = array_to_string(string_to_array(g_pos) * self._scale)
                element.set("pos", g_pos)
            if g_size is not None:
                g_size_np = string_to_array(g_size)
                # handle cases where size is not 3 dimensional
                if len(g_size_np) == 3:
                    g_size_np = g_size_np * self._scale
                elif len(g_size_np) == 2:
                    scale = np.array(self._scale).reshape(-1)
                    if len(scale) == 1:
                        g_size_np[1] *= scale
                    elif len(scale) == 3:
                        # g_size_np[0] *= np.mean(scale[:2])
                        g_size_np[0] *= np.mean(scale[:2])  # width
                        g_size_np[1] *= scale[2]  # height
                    else:
                        raise ValueError
                else:
                    raise ValueError
                g_size = array_to_string(g_size_np)
                element.set("size", g_size)

        # scale meshes
        meshes = self.asset.findall("mesh")
        for elem in meshes:
            m_scale = elem.get("scale")
            if m_scale is not None:
                m_scale = string_to_array(m_scale)
            else:
                m_scale = np.ones(3)

            m_scale *= self._scale
            elem.set("scale", array_to_string(m_scale))

        # scale bodies
        body_pairs = self._get_elements(obj, "body")
        for (_, elem) in body_pairs:
            b_pos = elem.get("pos")
            if b_pos is not None:
                b_pos = string_to_array(b_pos) * self._scale
                elem.set("pos", array_to_string(b_pos))

        # scale joints
        joint_pairs = self._get_elements(obj, "joint")
        for (_, elem) in joint_pairs:
            j_pos = elem.get("pos")
            if j_pos is not None:
                j_pos = string_to_array(j_pos) * self._scale
                elem.set("pos", array_to_string(j_pos))

        # scale sites
        site_pairs = self._get_elements(self.worldbody, "site")
        for (_, elem) in site_pairs:
            s_pos = elem.get("pos")
            if s_pos is not None:
                s_pos = string_to_array(s_pos) * self._scale
                elem.set("pos", array_to_string(s_pos))

            s_size = elem.get("size")
            if s_size is not None:
                s_size_np = string_to_array(s_size)
                # handle cases where size is not 3 dimensional
                if len(s_size_np) == 3:
                    s_size_np = s_size_np * self._scale
                elif len(s_size_np) == 2:
                    scale = np.array(self._scale).reshape(-1)
                    if len(scale) == 1:
                        s_size_np *= scale
                    elif len(scale) == 3:
                        s_size_np[0] *= np.mean(scale[:2])  # width
                        s_size_np[1] *= scale[2]  # height
                    else:
                        raise ValueError
                elif len(s_size_np) == 1:
                    s_size_np *= np.mean(self._scale)
                else:
                    raise ValueError
                s_size = array_to_string(s_size_np)
                elem.set("size", s_size)

    @property
    def bottom_offset(self):
        bottom_site = self.worldbody.find("./body/site[@name='{}bottom_site']".format(self.naming_prefix))
        return string_to_array(bottom_site.get("pos"))

    @property
    def top_offset(self):
        top_site = self.worldbody.find("./body/site[@name='{}top_site']".format(self.naming_prefix))
        return string_to_array(top_site.get("pos"))

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return string_to_array(horizontal_radius_site.get("pos"))[0]

    def get_bounding_box_half_size(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return string_to_array(horizontal_radius_site.get("pos")) - self.bottom_offset

    def _get_elements_by_name(self, geom_names, body_names=None, joint_names=None):
        """
        seaches for returns all geoms, bodies, and joints used for cabinet
        called by _get_cab_components, as implemented in subclasses

        for geoms, include both collision and visual geoms
        """

        # names of every geom
        geoms = {geom_name: list() for geom_name in geom_names}
        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": self.name + "_" + geom_name + postfix},
                    return_first=True,
                )
                geoms[geom_name].append(g)

        # get bodies
        bodies = dict()
        if body_names is not None:
            for body_name in body_names:
                bodies[body_name] = find_elements(
                    root=self._obj, tags="body", attribs={"name": self.name + "_" + body_name}, return_first=True
                )

        # get joints
        joints = dict()
        if joint_names is not None:
            for joint_name in joint_names:
                joints[joint_name] = find_elements(
                    root=self._obj, tags="joint", attribs={"name": self.name + "_" + joint_name}, return_first=True
                )
        return geoms, bodies, joints


class MujocoGeneratedObject(MujocoObject):
    """
    Base class for all procedurally generated objects.

    Args:
        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(self, obj_type="all", duplicate_collision_geoms=True):
        super().__init__(obj_type=obj_type, duplicate_collision_geoms=duplicate_collision_geoms)

        # Store common material names so we don't add prefixes to them
        self.shared_materials = set()
        self.shared_textures = set()

    def sanity_check(self):
        """
        Checks if data provided makes sense.
        Called in __init__()
        For subclasses to inherit from
        """
        pass

    @staticmethod
    def get_collision_attrib_template():
        """
        Generates template with collision attributes for a given geom

        Returns:
            dict: Initial template with `'pos'` and `'group'` already specified
        """
        return {"group": "0", "rgba": array_to_string(OBJECT_COLLISION_COLOR)}

    @staticmethod
    def get_visual_attrib_template():
        """
        Generates template with visual attributes for a given geom

        Returns:
            dict: Initial template with `'conaffinity'`, `'contype'`, and `'group'` already specified
        """
        return {"conaffinity": "0", "contype": "0", "mass": "1e-8", "group": "1"}

    def append_material(self, material):
        """
        Adds a new texture / material combination to the assets subtree of this XML
        Input is expected to be a CustomMaterial object

        See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
        Mujoco texture / material tags, respectively

        Note that the "file" attribute for the "texture" tag should be specified relative to the textures directory
        located in physics_simulator/models/assets/textures/

        Args:
            material (CustomMaterial): Material to add to this object
        """
        # First check if asset attribute exists; if not, define the asset attribute
        if not hasattr(self, "asset"):
            self.asset = ET.Element("asset")
        # If the material name is not in shared materials, add this to our assets
        if material.name not in self.shared_materials:
            self.asset.append(ET.Element("texture", attrib=material.tex_attrib))
            self.asset.append(ET.Element("material", attrib=material.mat_attrib))
        # Add this material name to shared materials if it should be shared
        if material.shared:
            self.shared_materials.add(material.name)
            self.shared_textures.add(material.tex_attrib["name"])
        # Update prefix for assets
        add_prefix(root=self.asset, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    def exclude_from_prefixing(self, inp):
        """
        Exclude all shared materials and their associated names from being prefixed.

        Args:
            inp (ET.Element or str): Element or its attribute to check for prefixing.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        # Automatically return False if this is not of type "str"
        if type(inp) is not str:
            return False
        # Only return True if the string matches the name of a common material
        return True if inp in self.shared_materials or inp in self.shared_textures else False

    # Methods that still need to be defined by subclass
    def _get_object_subtree(self):
        raise NotImplementedError

    def bottom_offset(self):
        raise NotImplementedError

    def top_offset(self):
        raise NotImplementedError

    def horizontal_radius(self):
        raise NotImplementedError

    def get_bounding_box_half_size(self):
        return np.array([self.horizontal_radius, self.horizontal_radius, 0.0]) - self.bottom_offset


class PrimitiveObject(MujocoGeneratedObject):
    """
    Base class for all programmatically generated mujoco object
    i.e., every MujocoObject that does not have an corresponding xml file

    Args:
        name (str): (unique) name to identify this generated object

        size (n-tuple of float): relevant size parameters for the object, should be of size 1 - 3

        rgba (4-tuple of float): Color

        density (float): Density

        friction (3-tuple of float): (sliding friction, torsional friction, and rolling friction).
            A single float can also be specified, in order to set the sliding friction (the other values) will
            be set to the MuJoCo default. See http://www.mujoco.org/book/modeling.html#geom for details.

        solref (2-tuple of float): MuJoCo solver parameters that handle contact.
            See http://www.mujoco.org/book/XMLreference.html for more details.

        solimp (3-tuple of float): MuJoCo solver parameters that handle contact.
            See http://www.mujoco.org/book/XMLreference.html for more details.

        material (CustomMaterial or `'default'` or None): if "default", add a template material and texture for this
            object that is used to color the geom(s).
            Otherwise, input is expected to be a CustomMaterial object

            See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
            Mujoco texture / material tags, respectively

            Note that specifying a custom texture in this way automatically overrides any rgba values set

        joints (None or str or list of dict): Joints for this object. If None, no joint will be created. If "default",
            a single (free) joint will be crated. Else, should be a list of dict, where each dictionary corresponds to
            a joint that will be created for this object. The dictionary should specify the joint attributes
            (type, pos, etc.) according to the MuJoCo xml specification.

        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(
        self,
        name,
        size=None,
        rgba=None,
        density=None,
        friction=None,
        solref=None,
        solimp=None,
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        # Always call superclass first
        super().__init__(obj_type=obj_type, duplicate_collision_geoms=duplicate_collision_geoms)

        # Set name
        self._name = name

        if size is None:
            size = [0.05, 0.05, 0.05]
        self.size = list(size)

        if rgba is None:
            rgba = [1, 0, 0, 1]
        assert len(rgba) == 4, "rgba must be a length 4 array"
        self.rgba = list(rgba)

        if density is None:
            density = 1000  # water
        self.density = density

        if friction is None:
            friction = [1, 0.005, 0.0001]  # MuJoCo default
        elif isinstance(friction, float) or isinstance(friction, int):
            friction = [friction, 0.005, 0.0001]
        assert len(friction) == 3, "friction must be a length 3 array or a single number"
        self.friction = list(friction)

        if solref is None:
            self.solref = [0.001, 1.0]  # MuJoCo default
        else:
            self.solref = solref

        if solimp is None:
            self.solimp = [0.998, 0.998, 0.001]  # MuJoCo default
        else:
            self.solimp = solimp

        self.material = material
        if material == "default":
            # add in default texture and material for this object (for domain randomization)
            default_tex = CustomMaterial(
                texture=self.rgba,
                tex_name="tex",
                mat_name="mat",
            )
            self.append_material(default_tex)
        elif material is not None:
            # add in custom texture and material
            self.append_material(material)

        # joints for this object
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints have names!
        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(i)

        # Always run sanity check
        self.sanity_check()

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def _get_object_subtree_(self, ob_type="box"):
        # Create element tree
        obj = new_body(name="main")

        # Get base element attributes
        element_attr = {"name": "g0", "type": ob_type, "size": array_to_string(self.size)}

        # Add collision geom if necessary
        if self.obj_type in {"collision", "all"}:
            col_element_attr = deepcopy(element_attr)
            col_element_attr.update(self.get_collision_attrib_template())
            col_element_attr["density"] = str(self.density)
            col_element_attr["friction"] = array_to_string(self.friction)
            col_element_attr["solref"] = array_to_string(self.solref)
            col_element_attr["solimp"] = array_to_string(self.solimp)
            obj.append(new_geom(**col_element_attr))
        # Add visual geom if necessary
        if self.obj_type in {"visual", "all"}:
            vis_element_attr = deepcopy(element_attr)
            vis_element_attr.update(self.get_visual_attrib_template())
            vis_element_attr["name"] += "_vis"
            if self.material == "default":
                vis_element_attr["rgba"] = "0.5 0.5 0.5 1"  # mujoco default
                vis_element_attr["material"] = "mat"
            elif self.material is not None:
                vis_element_attr["material"] = self.material.mat_attrib["name"]
            else:
                vis_element_attr["rgba"] = array_to_string(self.rgba)
            obj.append(new_geom(**vis_element_attr))
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # add a site as well
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["name"] = "default_site"
        obj.append(new_site(**site_element_attr))
        return obj

    # Methods that still need to be defined by subclass
    def _get_object_subtree(self):
        raise NotImplementedError

    def bottom_offset(self):
        raise NotImplementedError

    def top_offset(self):
        raise NotImplementedError

    def horizontal_radius(self):
        raise NotImplementedError