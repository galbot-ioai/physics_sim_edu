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
# Description: Example for creating light configuration
#
# Author: Herman Ye@Galbot
# Date: 2025-04-18
#
# This example demonstrates:
# 1. How to create light configuration classes using Pydantic
# 2. How to save and load configurations
# 3. How to handle configuration validation and type checking
#
#####################################################################################


from pydantic import BaseModel, Field, ConfigDict
from synthnova_config.utils import (
    save_config,
    load_config,
    delete_config,
    create_temp_config_file,
)
from synthnova_config import LightConfig, RectLight
import uuid


def main():
    """Main function demonstrating configuration creation, saving, loading, and deletion"""

    # Create a default light config instance
    default_config = LightConfig(
        light=RectLight(
            translation=[0, 0, 0], rotation=[0, 90, 0], width=1.0, height=1.0
        )
    )
    print("\nDefault configuration:")
    print(default_config)
    print("\nDefault configuration as JSON:")
    print(default_config.model_dump_json(indent=4))

    # Create a specific light config instance
    specific_prim_path = "/light/rect/123"
    specific_uuid = str(uuid.uuid4())
    specific_config = LightConfig(
        prim_path=specific_prim_path,
        uuid=specific_uuid,
        light=RectLight(
            translation=[0, 0, 0],
            rotation=[0, 0, 90],
            width=1.0,
            height=1.0,
        ),
    )

    print("\nSpecific configuration:")
    print(specific_config)
    print("\nSpecific configuration as JSON:")
    print(specific_config.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
