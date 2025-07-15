from physics_simulator.object import MujocoXML
from physics_simulator.utils.mjcf_utils import convert_to_string, find_elements
from physics_simulator.object import MujocoObject
import os

class MujocoWorld(MujocoXML):
    """A MuJoCo world representation as an XML model.
    
    This class provides a high-level interface for creating and manipulating 
    a MuJoCo simulation world. It inherits from MujocoXML and adds specific 
    functionality for managing world attributes and objects.
    """

    def __init__(self):
        """Initialize a new MuJoCo world from the base XML template."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(os.path.join(current_dir, "base.xml"))
        self.options = find_elements(
            root=self.root, tags="option", attribs=None, return_first=True
        )
        self.compiler = find_elements(
            root=self.root, tags="compiler", attribs=None, return_first=True
        )
        self.size = find_elements(
            root=self.root, tags="size", attribs=None, return_first=True
        )

    def __str__(self):
        """Return the XML string representation of the world."""
        return self.get_xml()

    def update_attribute(self, key, value, tag="option"):
        """Update a specific attribute in the world configuration.
        
        Args:
            key: Name of the attribute to update
            value: New value for the attribute
            tag: Section to update ("option", "compiler", or "size")
            
        Raises:
            ValueError: If the tag is not supported
        """
        if tag == "option":
            self.options.set(key, convert_to_string(value))
        elif tag == "compiler":
            self.compiler.set(key, convert_to_string(value))
        elif tag == "size":
            self.size.set(key, convert_to_string(value))
        else:
            raise ValueError(f"Invalid tag: {tag}")

    # Reference from class `Task`: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/tasks/task.py
    def merge_objects(self, mujoco_objects):
        """Add object models to the world.
        
        Merges the provided MuJoCo objects into this world model,
        including both their assets and physical representations.
        
        Args:
            mujoco_objects: List of MujocoObject instances to add to the world
            
        Raises:
            AssertionError: If any item is not a MujocoObject instance
        """
        for mujoco_obj in mujoco_objects:
            # Make sure we actually got a MujocoObject
            assert isinstance(
                mujoco_obj, MujocoObject
            ), "Tried to merge non-MujocoObject! Got type: {}".format(type(mujoco_obj))
            # Merge this object
            self.merge_assets(mujoco_obj)
            self.worldbody.append(mujoco_obj.get_obj())
