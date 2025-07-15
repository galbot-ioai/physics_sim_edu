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
"""IO utility functions for Synthnova configuration files.

This module provides convenient functions for handling configuration files,
including creating temporary files, saving, loading, and deleting configurations.
It supports both dictionary-based and class-based configuration management with
JSON serialization.

Functions:
    create_temp_config_file: Create a temporary configuration file
    save_config: Save configuration to a JSON file
    load_config: Load configuration from a JSON file
    delete_config: Delete a configuration file

Author: Herman Ye@Galbot
Date: 2025-04-16
"""

import json
import tempfile
from typing import Any, Dict, Optional, Union
from pathlib import Path


def create_temp_config_file(suffix: str = ".json") -> str:
    """Create a temporary configuration file with specified suffix.

    This function creates a temporary file that persists on disk until explicitly
    deleted. The file is created with the specified suffix and can be used for
    temporary configuration storage.

    Args:
        suffix (str, optional): File extension for the temporary file. Defaults to ".json".

    Returns:
        str: Absolute path to the created temporary file.

    Example:
        >>> temp_path = create_temp_config_file(".yaml")
        >>> print(temp_path)
        '/tmp/tmp8fn7_w3t.yaml'
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        return temp_file.name


def save_config(
    config: Union[Any, Dict[str, Any]],
    filepath: Union[str, Path],
    indent: int = 4,
) -> None:
    """Save configuration to a JSON file.

    This function saves a configuration object or dictionary to a JSON file. If the
    configuration object has a model_dump_json method (like Pydantic models), it will
    be used for serialization. Otherwise, json.dumps will be used for serialization.
    The function creates any necessary parent directories.

    Args:
        config (Union[Any, Dict[str, Any]]): Configuration object or dictionary to save.
            Must be JSON-serializable or have a model_dump_json method.
        filepath (Union[str, Path]): Path where to save the configuration file.
            Can be either a string path or a Path object.
        indent (int, optional): Number of spaces for JSON indentation. Defaults to 4.

    Raises:
        AttributeError: If config object doesn't have required serialization method.
        OSError: If there are permission issues or other I/O related errors.
        TypeError: If the object is not JSON serializable.

    Example:
        >>> config = {"server": "localhost", "port": 8080}
        >>> save_config(config, "server_config.json")
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Handle different types of config objects
    if hasattr(config, "model_dump_json"):
        # For Pydantic models
        json_str = config.model_dump_json(indent=indent)
    else:
        # For dictionaries and other JSON-serializable objects
        json_str = json.dumps(config, indent=indent, ensure_ascii=False)

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json_str)


def load_config(
    filepath: Union[str, Path], config_class: Optional[type] = None
) -> Union[Any, Dict[str, Any]]:
    """Load configuration from a JSON file.

    This function loads and optionally validates a configuration from a JSON file.
    If a config_class is provided, the loaded data will be validated and instantiated
    as an object of that class.

    Args:
        filepath (Union[str, Path]): Path to the configuration file.
            Can be either a string path or a Path object.
        config_class (Optional[type], optional): Class to validate and instantiate
            the config with. If None, returns the raw dictionary. Defaults to None.

    Returns:
        Union[Any, Dict[str, Any]]: Loaded configuration object or dictionary.
            If config_class is provided, returns an instance of that class.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValidationError: If the data doesn't match the config_class schema.

    Example:
        >>> # Load as dictionary
        >>> config = load_config("server_config.json")
        >>> # Load with validation class
        >>> config = load_config("server_config.json", ServerConfig)
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        config_dict = json.load(f)

    if config_class is not None:
        return config_class.model_validate(config_dict)
    return config_dict


def delete_config(filepath: Union[str, Path]) -> None:
    """Delete a configuration file.

    This function safely deletes a configuration file from the filesystem.
    It checks for file existence before attempting deletion.

    Args:
        filepath (Union[str, Path]): Path to the configuration file to delete.
            Can be either a string path or a Path object.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        OSError: If there are permission issues or other I/O related errors.

    Example:
        >>> delete_config("old_config.json")
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    filepath.unlink()
