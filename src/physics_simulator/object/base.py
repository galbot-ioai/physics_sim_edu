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
# Description: Base classes for Mujoco XML models and objects
# Author: Chenyu Cao@Galbot
# Date: 2025-04-01
#
# Note: This code is adapted from robosuite (https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/base.py)
# under the MIT License. Original copyright:
# Copyright (c) 2018-2022 ARISE Initiative
#####################################################################################

import io
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import mujoco
from physics_simulator.utils import XMLError
from physics_simulator.utils.mjcf_utils import (
    _element_filter,
    add_prefix,
    find_elements,
    recolor_collision_geoms,
    sort_elements,
    string_to_array,
)


class MujocoXML(object):
    """Base class for handling and manipulating MuJoCo XML models.
    
    Provides functionality for loading, merging, and modifying XML model files
    for use with MuJoCo physics simulation. The class maintains references to 
    key model elements like worldbody, actuator, asset, etc.
    """

    def __init__(self, fname):
        """Initialize a MuJoCo XML model from a file.
        
        Args:
            fname (str): Path to the MJCF XML file
        """
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.tendon = self.create_default_element("tendon")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")

        # Parse any default classes and replace them inline
        default = self.create_default_element("default")
        default_classes = self._get_default_classes(default)
        self._replace_defaults_inline(default_dic=default_classes)

        # Remove original default classes
        self.root.remove(default)

        self.resolve_asset_dependency()

    def resolve_asset_dependency(self):
        """Convert all file dependencies to absolute paths.
        
        This ensures that when models are merged, asset references remain valid.
        """

        for node in self.asset.findall("./*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)

    def create_default_element(self, name):
        """Create a XML element with the given name if it doesn't exist.
        
        Args:
            name (str): Name of the element to create
            
        Returns:
            ET.Element: The found or newly created element
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def merge(self, others, merge_body="default"):
        """Merge one or more MujocoXML models into this one.
        
        Args:
            others: One or more MujocoXML instances to merge
            merge_body: Where to merge the worldbody contents:
                - "default": Merge into the root worldbody
                - None: Don't merge worldbody contents
                - Any string: Merge into a body with this name
                
        Raises:
            XMLError: If others is not a MujocoXML instance
        """
        if type(others) is not list:
            others = [others]
        for idx, other in enumerate(others):
            if not isinstance(other, MujocoXML):
                raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
            if merge_body is not None:
                root = (
                    self.worldbody
                    if merge_body == "default"
                    else find_elements(
                        root=self.worldbody, tags="body", attribs={"name": merge_body}, return_first=True
                    )
                )
                for body in other.worldbody:
                    root.append(body)
            self.merge_assets(other)
            for one_actuator in other.actuator:
                self.actuator.append(one_actuator)
            for one_sensor in other.sensor:
                self.sensor.append(one_sensor)
            for one_tendon in other.tendon:
                self.tendon.append(one_tendon)
            for one_equality in other.equality:
                self.equality.append(one_equality)
            for one_contact in other.contact:
                self.contact.append(one_contact)


    def get_model(self, mode="mujoco"):
        """Generate a MuJoCo model from the current XML tree.
        
        Args:
            mode (str): Mode to interpret the XML (currently only "mujoco" supported)
            
        Returns:
            MjModel: MuJoCo model instance
            
        Raises:
            ValueError: If mode is not supported
        """

        available_modes = ["mujoco"]
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            if mode == "mujoco":
                model = mujoco.MjModel.from_xml_string(string.getvalue())
                return model
            raise ValueError("Unkown model mode: {}. Available options are: {}".format(mode, ",".join(available_modes)))

    def get_xml(self):
        """Get the XML tree as a string.
        
        Returns:
            str: String representation of the XML
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, pretty=False):
        """Save the XML model to a file.
        
        Args:
            fname (str): Output file path
            pretty (bool): Whether to format the XML for readability
        """
        with open(fname, "w") as f:
            xml_str = ET.tostring(self.root, encoding="unicode")
            if pretty:
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl="")
            f.write(xml_str)

    def merge_assets(self, other):
        """Merge assets from another MujocoXML instance into this one.
        
        Only adds assets that don't already exist in this model (based on name).
        
        Args:
            other: Another MujocoXML instance whose assets will be merged
        """
        for asset in other.asset:
            if (
                find_elements(root=self.asset, tags=asset.tag, attribs={"name": asset.get("name")}, return_first=True)
                is None
            ):
                self.asset.append(asset)

    def get_element_names(self, root, element_type):
        """Find all element names of a specific type in the model tree.
        
        Recursively searches through the XML tree for elements of the specified type.
        
        Args:
            root: Root element to start the search from
            element_type: Type of element to search for (e.g., "site", "geom")
            
        Returns:
            list: Names of all found elements of the specified type
        """
        names = []
        for child in root:
            if child.tag == element_type:
                names.append(child.get("name"))
            names += self.get_element_names(child, element_type)
        return names

    @staticmethod
    def _get_default_classes(default):
        """Extract default classes from the XML model.
        
        Creates a nested dictionary structure representing the default classes
        and their attributes.
        
        Args:
            default: The default tag XML element
            
        Returns:
            dict: Nested dictionary of default classes and their attributes
        """
        # Create nested dict to return
        default_dic = {}
        # Parse the default tag accordingly
        for cls in default:
            cls_name = cls.get("class")
            if cls_name is not None:
                default_dic[cls_name] = {}
                
                # Handle nested default classes
                for child in cls:
                    if child.tag == "default":
                        nested_cls_name = child.get("class")
                        if nested_cls_name is not None:
                            default_dic[nested_cls_name] = {c.tag: c for c in child}
                    else:
                        default_dic[cls_name][child.tag] = child
        return default_dic

    def _replace_defaults_inline(self, default_dic, root=None, inherited_childclass=None):
        """Apply default attributes to elements in the XML tree.
        
        Recursively processes the XML tree, replacing class references with
        their corresponding default attributes where not explicitly specified.
        Also handles childclass attributes that apply to all child elements.
        
        Args:
            default_dic: Dictionary of default classes and their attributes
            root: Root element to start processing from (None for top-level)
            inherited_childclass: Class to inherit from parent's childclass attribute
        """
        # If root is None, this is the top level call -- replace root with self.root
        if root is None:
            root = self.root
            
        # Handle inherited childclass from parent
        effective_class = root.attrib.get("class", inherited_childclass)
        
        # Remove class attribute after reading it
        cls_name = root.attrib.pop("class", None)
        
        # Handle childclass attribute
        childclass_name = root.attrib.pop("childclass", None)
        
        # Apply class attributes if present
        if effective_class is not None:
            # If the tag for this element is contained in our default dic, we add any defaults that are not
            # explicitly specified in this
            tag_attrs = default_dic.get(effective_class, {}).get(root.tag, None)
            if tag_attrs is not None:
                for k, v in tag_attrs.attrib.items():
                    if root.get(k, None) is None:
                        root.set(k, v)
        
        # Loop through all child elements, passing childclass to children
        for child in root:
            self._replace_defaults_inline(default_dic=default_dic, root=child, inherited_childclass=childclass_name)

    @property
    def name(self):
        """
        Returns name of this MujocoXML

        Returns:
            str: Name of this MujocoXML
        """
        return self.root.get("model")


class MujocoModel(object):
    """Base interface for MuJoCo models defining standard properties and methods.
    
    This abstract class provides a consistent interface for accessing model components
    such as bodies, joints, actuators, and geometries. It also defines methods for
    name correction and site visibility control.
    
    Subclasses must implement various abstract properties including name, bodies,
    joints, and geometric properties.
    """

    def correct_naming(self, names):
        """Add the naming prefix to names unless explicitly excluded.
        
        Args:
            names: Name or collection of names to be corrected (string, list, or dict)
            
        Returns:
            Same type as input with prefixes added to names
            
        Raises:
            TypeError: If names is not a string, list, or dictionary
        """
        if type(names) is str:
            return self.naming_prefix + names if not self.exclude_from_prefixing(names) else names
        elif type(names) is list:
            return [self.naming_prefix + name if not self.exclude_from_prefixing(name) else name for name in names]
        elif type(names) is dict:
            names = names.copy()
            for key, val in names.items():
                names[key] = self.correct_naming(val)
            return names
        else:
            # Assumed to be type error
            raise TypeError("Error: type of 'names' must be str, list, or dict!")

    def set_sites_visibility(self, sim, visible):
        """Set the visibility of all sites in the model.
        
        Args:
            sim: MuJoCo simulation instance
            visible: If True, sites will be visible; if False, sites will be hidden
        """
        # Loop through all visualization geoms and set their alpha values appropriately
        for vis_g in self.sites:
            vis_g_id = sim.model.site_name2id(vis_g)
            if (visible and sim.model.site_rgba[vis_g_id][3] < 0) or (
                not visible and sim.model.site_rgba[vis_g_id][3] > 0
            ):
                # We toggle the alpha value
                sim.model.site_rgba[vis_g_id][3] = -sim.model.site_rgba[vis_g_id][3]

    def exclude_from_prefixing(self, inp):
        """
        A function that should take in an arbitrary input and return either True or False, determining whether the
        corresponding name to @inp should have naming_prefix added to it. Must be defined by subclass.

        Args:
            inp (any): Arbitrary input, depending on subclass. Can be str, ET.Element, etc.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Name for this model. Should be unique.

        Returns:
            str: Unique name for this model.
        """
        raise NotImplementedError

    @property
    def naming_prefix(self):
        """
        Generates a standardized prefix to prevent naming collisions

        Returns:
            str: Prefix unique to this model.
        """
        raise NotImplementedError

    @property
    def root_body(self):
        """
        Root body name for this model. This should correspond to the top-level body element in the equivalent mujoco xml
        tree for this object.
        """
        raise NotImplementedError

    @property
    def bodies(self):
        """
        Returns:
            list: Body names for this model
        """
        raise NotImplementedError

    @property
    def joints(self):
        """
        Returns:
            list: Joint names for this model
        """
        raise NotImplementedError

    @property
    def actuators(self):
        """
        Returns:
            list: Actuator names for this model
        """
        raise NotImplementedError

    @property
    def sites(self):
        """
        Returns:
             list: Site names for this model
        """
        raise NotImplementedError

    @property
    def sensors(self):
        """
        Returns:
             list: Sensor names for this model
        """
        raise NotImplementedError

    @property
    def contact_geoms(self):
        """
        List of names corresponding to the geoms used to determine contact with this model.

        Returns:
            list: relevant contact geoms for this model
        """
        raise NotImplementedError

    @property
    def visual_geoms(self):
        """
        List of names corresponding to the geoms used for visual rendering of this model.

        Returns:
            list: relevant visual geoms for this model
        """
        raise NotImplementedError

    @property
    def important_geoms(self):
        """
        Geoms corresponding to important components of this model. String keywords should be mapped to lists of geoms.

        Returns:
            dict of list: Important set of geoms, where each set of geoms are grouped as a list and are
            organized by keyword string entries into a dict
        """
        raise NotImplementedError

    @property
    def important_sites(self):
        """
        Dict of sites corresponding to the important site geoms (e.g.: used to aid visualization during sim).

        Returns:
            dict: Important site geoms, where each specific geom name is mapped from keyword string entries
                in the dict
        """
        raise NotImplementedError

    @property
    def important_sensors(self):
        """
        Dict of important sensors enabled for this model.

        Returns:
            dict: Important sensors for this model, where each specific sensor name is mapped from keyword string
                entries in the dict
        """
        raise NotImplementedError

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


class MujocoXMLModel(MujocoXML, MujocoModel):
    """A combined class that implements both the XML parsing and model interface.
    
    Provides functionality for loading, manipulating, and accessing MuJoCo models
    through both the XML representation and the model interface.
    """

    def __init__(self, fname, idn=0):
        """Initialize a MuJoCo XML model with an identifier.
        
        Args:
            fname (str): Path to the MJCF XML file
            idn (int): Identifier for the model, useful for prefixing
        """
        super().__init__(fname)

        # Set id and add prefixes to all body names to prevent naming clashes
        self.idn = idn

        # Define other variables that get filled later
        self.mount = None

        # Define filter method to automatically add a default name to visual / collision geoms if encountered
        group_mapping = {
            None: "col",
            "0": "col",
            "1": "vis",
        }
        ctr_mapping = {
            "col": 0,
            "vis": 0,
        }

        def _add_default_name_filter(element, parent):
            # Run default filter
            filter_key = _element_filter(element=element, parent=parent)
            # Also additionally modify element if it is (a) a geom and (b) has no name
            if element.tag == "geom" and element.get("name") is None:
                group = group_mapping[element.get("group")]
                element.set("name", f"g{ctr_mapping[group]}_{group}")
                ctr_mapping[group] += 1
            # Return default filter key
            return filter_key

        # Parse element tree to get all relevant bodies, joints, actuators, and geom groups
        self._elements = sort_elements(root=self.root, element_filter=_add_default_name_filter)
        assert (
            len(self._elements["root_body"]) == 1
        ), "Invalid number of root bodies found for robot model. Expected 1," "got {}".format(
            len(self._elements["root_body"])
        )
        self._elements["root_body"] = self._elements["root_body"][0]
        self._elements["bodies"] = (
            [self._elements["root_body"]] + self._elements["bodies"]
            if "bodies" in self._elements
            else [self._elements["root_body"]]
        )
        self._root_body = self._elements["root_body"].get("name")
        self._bodies = [e.get("name") for e in self._elements.get("bodies", [])]
        self._joints = [e.get("name") for e in self._elements.get("joints", [])]
        self._actuators = [e.get("name") for e in self._elements.get("actuators", [])]
        self._sites = [e.get("name") for e in self._elements.get("sites", [])]
        self._sensors = [e.get("name") for e in self._elements.get("sensors", [])]
        self._contact_geoms = [e.get("name") for e in self._elements.get("contact_geoms", [])]
        self._visual_geoms = [e.get("name") for e in self._elements.get("visual_geoms", [])]
        self._base_offset = string_to_array(self._elements["root_body"].get("pos", "0 0 0"))

        # Update all xml element prefixes
        add_prefix(root=self.root, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

        # Recolor all collision geoms appropriately
        recolor_collision_geoms(root=self.worldbody, rgba=self.contact_geom_rgba)


    def exclude_from_prefixing(self, inp):
        """
        By default, don't exclude any from being prefixed
        """
        return False

    @property
    def base_offset(self):
        """
        Provides position offset of root body.

        Returns:
            3-array: (x,y,z) pos value of root_body body element. If no pos in element, returns all zeros.
        """
        return self._base_offset

    @property
    def name(self):
        return "{}{}".format(type(self).__name__, self.idn)

    @property
    def naming_prefix(self):
        return "{}_".format(self.idn)

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
    def important_sites(self):
        return self.correct_naming(self._important_sites)

    @property
    def important_geoms(self):
        return self.correct_naming(self._important_geoms)

    @property
    def important_sensors(self):
        return self.correct_naming(self._important_sensors)

    @property
    def _important_sites(self):
        """
        Dict of sites corresponding to the important site geoms (e.g.: used to aid visualization during sim).

        Returns:
            dict: Important site geoms, where each specific geom name is mapped from keyword string entries
                in the dict. Note that the mapped sites should be the RAW site names found directly in the XML file --
                the naming prefix will be automatically added in the public method call
        """
        raise NotImplementedError

    @property
    def _important_geoms(self):
        """
        Geoms corresponding to important components of this model. String keywords should be mapped to lists of geoms.

        Returns:
            dict of list: Important set of geoms, where each set of geoms are grouped as a list and are
                organized by keyword string entries into a dict. Note that the mapped geoms should be the RAW geom
                names found directly in the XML file -- the naming prefix will be automatically added in the
                public method call
        """
        raise NotImplementedError

    @property
    def _important_sensors(self):
        """
        Dict of important sensors enabled for this model.

        Returns:
            dict: Important sensors for this model, where each specific sensor name is mapped from keyword string
                entries in the dict. Note that the mapped geoms should be the RAW sensor names found directly in the
                XML file -- the naming prefix will be automatically added in the public method call
        """
        raise NotImplementedError

    @property
    def contact_geom_rgba(self):
        """
        RGBA color to assign to all contact geoms for this model

        Returns:
            4-array: (r,g,b,a) values from 0 to 1 for this model's set of contact geoms
        """
        raise NotImplementedError

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        Useful for, e.g. placing models on a surface.
        By default, this corresponds to the root_body's base offset.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        return self.base_offset

    @property
    def top_offset(self):
        raise NotImplementedError

    @property
    def horizontal_radius(self):
        raise NotImplementedError
