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
# Description: Path manager for SynthNova Physics Simulator
# Author: Chenyu Cao@Galbot
# Date: 2025-04-02
#
#####################################################################################


from pathlib import Path


class PathManager:
    """Utility for managing file paths within the Physics Simulator project.
    
    Provides a set of static methods for accessing directories and files within
    the project structure, maintaining consistent paths across the application.
    All methods can be used without instantiation as they are class methods.
    """

    _root_dir = None

    @classmethod
    def _get_root_dir(cls) -> Path:
        """Get and cache the project's root directory path.
        
        Determines the root directory by traversing up from the current file's location.
        The path is cached after first retrieval for better performance.
        
        Returns:
            Path: Absolute path to the project's root directory
        """
        if cls._root_dir is None:
            cls._root_dir = Path(__file__).parent.parent.parent.parent
        return cls._root_dir

    @classmethod
    def get_root_path(cls, relative_path: str = None) -> str:
        """Get the absolute path to the root directory or a file within it.
        
        Args:
            relative_path: Optional path relative to the root directory
            
        Returns:
            str: Absolute path to the root directory or specified file
        """
        return str(
            cls._get_root_dir() / relative_path
            if relative_path
            else cls._get_root_dir()
        )

    @classmethod
    def get_asset_path(cls, relative_path: str = None) -> str:
        """Get the absolute path to the assets directory or a file within it.
        
        Args:
            relative_path: Optional path relative to the assets directory
            
        Returns:
            str: Absolute path to the assets directory or specified asset file
        """
        return str(
            cls._get_root_dir() / "assets" / relative_path
            if relative_path
            else cls._get_root_dir() / "assets"
        )
        
    @classmethod
    def get_robosuite_asset_path(cls, relative_path: str = None) -> str:
        """Get the absolute path to the robosuite assets directory or a file within it.
        
        Args:
            relative_path: Optional path relative to the robosuite assets directory
            
        Returns:
            str: Absolute path to the robosuite assets directory or specified file
        """
        return str(
            cls._get_root_dir() / "assets" / "robosuite" / relative_path
            if relative_path
            else cls._get_root_dir() / "assets" / "robosuite"
        )
        
    @classmethod
    def get_config_path(cls, relative_path: str = None) -> str:
        """Get the absolute path to the config directory or a file within it.
        
        Args:
            relative_path: Optional path relative to the config directory
            
        Returns:
            str: Absolute path to the config directory or specified configuration file
        """
        return str(
            cls._get_root_dir() / "config" / relative_path
            if relative_path
            else cls._get_root_dir() / "config"
        )