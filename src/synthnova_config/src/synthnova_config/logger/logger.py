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
# Description: Logger configuration module for Synthnova
# Author: Herman Ye@Galbot
# Date: 2025-04-16
#
#####################################################################################

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Dict, Any


class LoggerConfig(BaseModel):
    """Configuration class for managing logging behavior in the Synthnova application.

    This class provides a structured way to configure logging parameters, including log levels,
    output destinations, and file paths. It uses Pydantic for data validation and serialization.

    The logging system supports multiple severity levels and can output to both console and file
    simultaneously. File logging is optional and can be configured with custom paths.

    Attributes:
        log_level (Literal["debug", "info", "warning", "error"]): 
            The minimum severity level for log messages to be processed.
            - debug: Detailed diagnostic information for debugging purposes
            - info: General operational information about program execution
            - warning: Indicates potential issues that don't prevent execution
            - error: Serious issues that may prevent normal program execution
            Defaults to "info".
        
        use_file_log (bool): 
            Controls whether logs are written to a file in addition to console output.
            - True: Logs are written to both console and file
            - False: Logs are written only to console
            Defaults to True.
        
        log_path (Optional[str]): 
            The absolute path to the directory where log files will be stored.
            - None: Uses the system's default logging directory
            - str: Custom directory path (will be created if it doesn't exist)
            Defaults to None.

    Example:
        Basic configuration with default settings:
        >>> logger_config = LoggerConfig()

        Custom configuration with specific settings:
        >>> logger_config = LoggerConfig(
        ...     log_level="debug",
        ...     use_file_log=True,
        ...     log_path="/var/log/synthnova"
        ... )
    """

    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Minimum severity level for log messages to be processed",
        json_schema_extra={
            "examples": ["debug", "info", "warning", "error"],
            "order": 1,
        },
    )

    use_file_log: bool = Field(
        default=True,
        description="Enable or disable logging to file in addition to console output",
        json_schema_extra={"examples": [True, False], "order": 2},
    )

    log_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the log file directory. If None, uses system default",
        json_schema_extra={"examples": ["/path/to/logs", None], "order": 3},
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Enable validation when attributes are assigned
        extra="forbid",  # Prevent additional fields not defined in the model
        frozen=False,  # Allow modification of fields after instance creation
        str_strip_whitespace=True,  # Remove leading and trailing whitespace from string values
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert LoggerConfig to a dictionary.

        Returns:
            Dictionary representation of LoggerConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggerConfig":
        """Create an LoggerConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            LoggerConfig: Configured logger instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
    