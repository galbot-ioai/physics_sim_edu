# Constants file for physics simulator
# This file centralizes all magic numbers and hardcoded values used throughout the codebase

# Default simulation parameters
DEFAULT_TIMESTEP = 0.001  # Default physics timestep in seconds
DEFAULT_SOLVER_ITERATIONS = 10  # Default number of solver iterations
DEFAULT_PHYSICS_FREQUENCY = 1000  # Hz

# Default camera parameters
DEFAULT_CAMERA_FOV = 45.0  # Default field of view in degrees
DEFAULT_CAMERA_WIDTH = 640  # Default camera width in pixels
DEFAULT_CAMERA_HEIGHT = 480  # Default camera height in pixels
DEFAULT_CAMERA_NEAR = 0.01  # Default near clipping plane
DEFAULT_CAMERA_FAR = 100.0  # Default far clipping plane

# Default render parameters
DEFAULT_RENDER_WIDTH = 1920
DEFAULT_RENDER_HEIGHT = 1080
DEFAULT_RENDER_DEVICE_ID = -1  # Use default GPU

# Physical constants
GRAVITY_EARTH = -9.81  # Earth gravity in m/s^2
DEFAULT_DENSITY = 1000.0  # Default object density in kg/m^3
DEFAULT_FRICTION = [1.0, 0.005, 0.0001]  # Default friction coefficients [sliding, torsional, rolling]

# Default geometric sizes
DEFAULT_BOX_SIZE = [0.1, 0.1, 0.1]  # Default box dimensions in meters
DEFAULT_SPHERE_RADIUS = 0.1  # Default sphere radius in meters
DEFAULT_CYLINDER_RADIUS = 0.1  # Default cylinder radius in meters
DEFAULT_CYLINDER_HEIGHT = 0.2  # Default cylinder height in meters

# Default colors (RGBA)
DEFAULT_OBJECT_COLOR = [0.8, 0.6, 0.4, 1.0]  # Default object color
DEFAULT_ROBOT_COLOR = [0.7, 0.7, 0.7, 1.0]  # Default robot color
DEFAULT_GROUND_COLOR = [0.5, 0.5, 0.5, 1.0]  # Default ground color

# Tolerances and thresholds
POSITION_TOLERANCE = 1e-6  # Position comparison tolerance
ORIENTATION_TOLERANCE = 1e-6  # Orientation comparison tolerance
VELOCITY_TOLERANCE = 1e-6  # Velocity comparison tolerance

# Limits
MAX_JOINT_VELOCITY = 10.0  # Maximum joint velocity in rad/s
MAX_JOINT_ACCELERATION = 100.0  # Maximum joint acceleration in rad/s^2
MAX_FORCE = 1000.0  # Maximum force in Newtons
MAX_TORQUE = 100.0  # Maximum torque in Nm

# Default paths
DEFAULT_ASSETS_DIR = "assets"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_LOGS_DIR = "logs"

# Thread and timing constants
DEFAULT_THREAD_TIMEOUT = 5.0  # Default timeout for thread operations in seconds
DEFAULT_SLEEP_TIME = 0.001  # Default sleep time in loops in seconds

# Error codes and return values
SUCCESS = 0
ERROR_INVALID_INPUT = -1
ERROR_SIMULATION_NOT_RUNNING = -2
ERROR_OBJECT_NOT_FOUND = -3
ERROR_ROBOT_NOT_FOUND = -4
ERROR_SENSOR_NOT_FOUND = -5

# String format constants
FLOAT_PRECISION = 6  # Number of decimal places for float formatting 