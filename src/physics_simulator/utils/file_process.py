import numpy as np
import cv2  # Ensure you have OpenCV installed
import auro_utils.manager as au
import tifffile as tiff


def write_jpg(data: np.ndarray, file_path: str) -> None:
    """
    Saves the RGB color data as a JPEG image.

    Args:
        data (np.ndarray):
            The RGB color data to save. Must have shape (H, W, 3).
        file_path (str):
            The absolute path or current cmd path of the file to which the image will be saved.
                        The file_path should include the .jpg extension.

    Raises:
        ValueError: If the RGB data has an unexpected shape or if the
                     file_path does not end with .jpg.
    """
    if data.shape[-1] != 3:
        raise ValueError("RGB data must have 3 channels (R, G, B).")

    if not file_path.lower().endswith(".jpg"):
        raise ValueError("File path must end with .jpg.")

    # Convert the RGB data to BGR format for OpenCV
    bgr_data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save the image
    au.ensure_path_exists(file_path)
    cv2.imwrite(file_path, bgr_data)


def read_jpg(file_path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads a JPEG image file and returns the RGB color data.

    Args:
        file_path (str):
            The absolute path or current cmd path of the JPEG file to read. The file_path
            should include the .jpg extension.
        rgb (bool):
            Whether to return the data in RGB format. If False,
            the data will be returned in BGR format.

    Returns:
        np.ndarray:
            A NumPy array of shape (H, W, 3) representing the RGB
            color data of the image, where H is the height and W
            is the width of the image.

    Raises:
        ValueError: If the file does not end with .jpg or if the image
                     cannot be read.
    """
    if not file_path.lower().endswith(".jpg"):
        raise ValueError("File path must end with .jpg.")

    # Read the image using OpenCV
    au.check_file_exists(file_path)
    bgr_data = cv2.imread(file_path)

    if bgr_data is None:
        raise ValueError(f"Unable to read the image from {file_path}.")

    # Convert the BGR data to RGB format
    if rgb:
        data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)

    return data


def write_tiff(data: np.ndarray, file_path: str, photometric="minisblack") -> None:
    """
    Saves data as a TIFF image.

    Args:
        data (np.ndarray):
            The data to save. Can be in various shapes including (H, W, 3) for RGB,
            (H, W) for grayscale, or (n, m, 1) for depth data.
        file_path (str):
            The absolute path or current cmd path of the file to which the image will be saved.
            The file_path should include the .tiff or .tif extension.
        photometric (str):
            The photometric interpretation of the TIFF image. Default is "minisblack".


    Raises:
        ValueError: If the data has an unexpected shape or if the
                     file_path does not end with .tiff or .tif.
    """

    if not (file_path.lower().endswith(".tiff") or file_path.lower().endswith(".tif")):
        raise ValueError("File path must end with .tiff or .tif.")

    # Ensure correct shape for depth data
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data.squeeze(-1)  # Remove the last dimension for depth data

    # Save the image
    au.ensure_path_exists(file_path)
    tiff.imwrite(file_path, data, photometric=photometric)  # Use specified dtype


def preprocess_depth(
    depth_data,
    scale=1,
    min_value=None,
    max_value=None,
    data_type: np.dtype = np.float32,
):
    """
    Preprocesses the depth data by scaling, clamping, and converting to a specified data type.

    Args:
        depth_data (np.ndarray):
            The raw depth data to process. Should be a 1D or 2D array.
        scale (float, optional):
            The factor by which to scale the depth data. Default is 1.
        min_value (float, optional):
            The minimum value to clamp the depth data. If None, no lower limit is applied.
        max_value (float, optional):
            The maximum value to clamp the depth data. If None, no upper limit is applied.
        data_type (np.dtype, optional):
            The desired data type for the processed data. E.g., np.uint16 or np.float32. Default is np.float32.

    Returns:
        np.ndarray:
            The processed depth data, scaled, clamped, and converted to the specified data type.

    Raises:
        ValueError: If the depth data is not a valid shape or if the specified min_value/max_value are invalid.
    """
    depth_data_scaled = depth_data * scale
    if min_value is not None or max_value is not None:
        depth_data_clamped = np.clip(depth_data_scaled, 
                                     min_value if min_value is not None else depth_data_scaled.min(),
                                     max_value if max_value is not None else depth_data_scaled.max())
    else:
        depth_data_clamped = depth_data_scaled
    processed_data = depth_data_clamped.astype(data_type)
    return processed_data


def colorize_depth(depth_data):
    """
    Convert depth data to a colorized RGB image for visualization.

    Args:
        depth_data: Depth data array

    Returns:
        Colorized RGB image
    """
    # Normalize to 8-bit for colormap application
    depth_normalized = cv2.normalize(
        depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Apply colormap
    depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    return depth_colorized
