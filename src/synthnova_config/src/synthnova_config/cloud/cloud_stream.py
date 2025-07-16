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
# Description: Synthnova Cloud Streaming Config
# Author: Herman Ye@Galbot
# Date: 2025-04-16
#
#####################################################################################
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal
from typing import Dict, Any
import ipaddress
import re


class CloudStreamingConfig(BaseModel):
    """Configuration class for Synthnova cloud streaming parameters.

    This class serves as the central configuration for cloud streaming settings in the
    Synthnova framework. It defines parameters for WebRTC streaming and
    other cloud streaming configurations.

    Attributes:
        webrtc_cloud_streaming (bool): Enable/disable Galbot WebRTC cloud streaming.
            When enabled, the system will use WebRTC protocol for real-time streaming.
            Default: False

        webrtc_host (str): The host address for WebRTC streaming server.
            Use "0.0.0.0" to listen on all network interfaces.
            Default: "0.0.0.0"

        webrtc_port (int): The port number for WebRTC streaming server.
            Must be between 1024 and 65535.
            Default: 8080

        nvidia_native_cloud_streaming (bool): Enable/disable NVIDIA Isaac Sim native cloud streaming.
            When enabled, the system will use NVIDIA's native streaming protocol.
            Default: False

        nvidia_webrtc_cloud_streaming (bool): Enable/disable NVIDIA Isaac Sim WebRTC cloud streaming.
            When enabled, the system will use WebRTC protocol for NVIDIA Isaac Sim.
            Default: False

    Example:
        ```python
        config = CloudStreamingConfig(
            webrtc_cloud_streaming=True,
            webrtc_host="0.0.0.0",
            webrtc_port=8080,
            nvidia_native_cloud_streaming=False,
            nvidia_webrtc_cloud_streaming=True
        )
        ```

    Note:
        - Only one streaming method should be enabled at a time to avoid conflicts.
        - WebRTC streaming requires proper network configuration and firewall settings.
        - NVIDIA streaming options require NVIDIA Isaac Sim to be properly installed and configured.
    """

    webrtc_cloud_streaming: bool = Field(
        default=False,
        description="Enable Galbot WebRTC cloud streaming",
        json_schema_extra={"examples": [True, False]},
    )

    webrtc_host: str = Field(
        default="0.0.0.0",
        description="Galbot WebRTC host",
        json_schema_extra={"examples": ["0.0.0.0", "127.0.0.1"]},
    )

    webrtc_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Galbot WebRTC port",
        json_schema_extra={"examples": [8080, 8081]},
    )

    nvidia_native_cloud_streaming: bool = Field(
        default=False,
        description="Enable NVIDIA Isaac Sim native cloud streaming",
        json_schema_extra={"examples": [True, False]},
    )
    nvidia_webrtc_cloud_streaming: bool = Field(
        default=False,
        description="Enable NVIDIA Isaac Sim WebRTC cloud streaming",
        json_schema_extra={"examples": [True, False]},
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Enable runtime validation of field assignments
        extra="forbid",  # Prevent accidental addition of undefined fields
        frozen=False,  # Allow modification of configuration after initialization
    )

    @field_validator("webrtc_host")
    @classmethod
    def validate_webrtc_host(cls, v: str) -> str:
        """Validate the webrtc_host field.

        Args:
            v: The host value to validate

        Returns:
            The validated host value

        Raises:
            ValueError: If the host value is invalid
        """
        # Allow "0.0.0.0" as a special case
        if v == "0.0.0.0":
            return v

        # Try to validate as IP address
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            pass

        # Validate as hostname
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$', v):
            raise ValueError(f"Invalid host value: {v}")

        # Check total length
        if len(v) > 255:
            raise ValueError(f"Host value too long: {v}")

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert ObjectConfig to a dictionary.

        Returns:
            Dictionary representation of CloudStreamingConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudStreamingConfig":
        """Create an CloudStreamingConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            CloudStreamingConfig: Configured cloud streaming instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
