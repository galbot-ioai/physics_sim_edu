# Pose Estimation

## Installation
To install the required packages, run:

```bash
# Install dependencies
conda create --name pose python=3.8
conda activate pose
pip install -r requirements.txt

# Install teaser
sudo apt update -y
sudo apt install cmake libeigen3-dev libboost-all-dev
cd TEASER-plusplus && mkdir build
cd build
find / -name "pybind11Config.cmake" 2>/dev/null # This will help you find the path to pybind11
cmake -DBUILD_PYTHON_BINDINGS=ON -DTEASERPP_PYTHON_VERSION=3.8 -Dpybind11_DIR=/path/to/pybind11 ..
# eg. cmake -DBUILD_PYTHON_BINDINGS=ON -DTEASERPP_PYTHON_VERSION=3.8 -Dpybind11_DIR=/home/xmfang/miniconda3/envs/pose/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 ..
make teaserpp_python
cd ..
pip install .
```

## Usage
An example usage of the pose estimation script is provided in `example.py`. You can run it as follows:
```bash
conda activate pose
python3 examply.py
```

## Debugging
To debug the pose estimation process, you can use the `debug.py` script. It will load the parameters from a specified debug directory and visualize the results. You can run it as follows:
```bash
python3 debug.py --debug_file_dir <debug_folder_path> --visualize
```