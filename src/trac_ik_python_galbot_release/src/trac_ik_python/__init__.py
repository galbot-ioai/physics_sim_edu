__version__ = "0.0.1"

import os

os.environ["LD_LIBRARY_PATH"] = os.path.dirname(__file__) + ":" + os.environ.get("LD_LIBRARY_PATH", "")