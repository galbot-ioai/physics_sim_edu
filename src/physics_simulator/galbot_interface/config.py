from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class ModulesManagerConfig(BaseModel):
    enabled_modules: List[str] = Field(
        default=[],
        description="The modules to be enabled",
    )

class ChassisConfig(BaseModel):
    joint_names: List[str] = Field(
        default=[
            "mobile_forward_joint",
            "mobile_side_joint",
            "mobile_yaw_joint",
        ],
        description="Names of the joints in the chassis",
    )


class RobotConfig(BaseModel):
    prim_path: str = Field(
        default="/World/my_robot",
        description="The prim path of the robot",
    )


class LeftArmConfig(BaseModel):
    joint_names: List[str] = Field(
        default=[
            "left_arm_joint1",
            "left_arm_joint2",
            "left_arm_joint3",
            "left_arm_joint4",
            "left_arm_joint5",
            "left_arm_joint6",
            "left_arm_joint7",
        ],
        description="Names of the joints in the left arm",
    )


class RightArmConfig(BaseModel):
    joint_names: List[str] = Field(
        default=[
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
            "right_arm_joint7",
        ],
        description="Names of the joints in the right arm",
    )


class HeadConfig(BaseModel):
    joint_names: List[str] = Field(
        default=["head_joint1", "head_joint2"],
        description="Names of the joints in the head",
    )


class LegConfig(BaseModel):
    joint_names: List[str] = Field(
        default=["leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4"],
        description="Names of the joints in the leg",
    )


class LeftGripperConfig(BaseModel):
    joint_names: List[str] = Field(
        default=[
            "left_gripper_joint1",
            "left_gripper_joint2",
            "left_gripper_joint3",
            "left_gripper_joint4",
            "left_gripper_joint5",
            "left_gripper_joint6",
        ],
        description="Names of the joints in the left gripper",
    )


class RightGripperConfig(BaseModel):
    joint_names: List[str] = Field(
        default=[
            "right_gripper_joint1",
            "right_gripper_joint2",
            "right_gripper_joint3",
            "right_gripper_joint4",
            "right_gripper_joint5",
            "right_gripper_joint6",
        ],
        description="Names of the joints in the right gripper",
    )


class LeftWristCameraConfig(BaseModel):
    prim_path_rgb: Optional[str] = Field(
        default=None,
        description="The prim path of the left wrist rgb camera",
    )

    prim_path_depth: Optional[str] = Field(
        default=None,
        description="The prim path of the left wrist depth camera",
    )


class RightWristCameraConfig(BaseModel):
    prim_path_rgb: Optional[str] = Field(
        default=None,
        description="The prim path of the right wrist rgb camera",
    )

    prim_path_depth: Optional[str] = Field(
        default=None,
        description="The prim path of the right wrist depth camera",
    )


class FrontHeadCameraConfig(BaseModel):
    prim_path_rgb: Optional[str] = Field(
        default=None,
        description="The prim path of the front head rgb camera",
    )

    prim_path_depth: Optional[str] = Field(
        default=None,
        description="The prim path of the front head depth camera",
    )


class GalbotInterfaceConfig(BaseModel):
    modules_manager: ModulesManagerConfig = Field(
        default=ModulesManagerConfig(),
        description="Configuration for the module manager",
    )

    robot: RobotConfig = Field(
        default=RobotConfig(),
        description="Configuration for the robot",
    )

    left_arm: LeftArmConfig = Field(
        default=LeftArmConfig(),
        description="Configuration for the left arm",
    )

    right_arm: RightArmConfig = Field(
        default=RightArmConfig(),
        description="Configuration for the right arm",
    )

    head: HeadConfig = Field(
        default=HeadConfig(),
        description="Configuration for the head",
    )

    leg: LegConfig = Field(
        default=LegConfig(),
        description="Configuration for the leg",
    )

    left_gripper: LeftGripperConfig = Field(
        default=LeftGripperConfig(),
        description="Configuration for the left gripper",
    )

    right_gripper: RightGripperConfig = Field(
        default=RightGripperConfig(),
        description="Configuration for the right gripper",
    )

    left_wrist_camera: LeftWristCameraConfig = Field(
        default=LeftWristCameraConfig(),
        description="Configuration for the left wrist camera",
    )

    right_wrist_camera: RightWristCameraConfig = Field(
        default=RightWristCameraConfig(),
        description="Configuration for the right wrist camera",
    )

    front_head_camera: FrontHeadCameraConfig = Field(
        default=FrontHeadCameraConfig(),
        description="Configuration for the front head camera",
    )

    chassis: ChassisConfig = Field(
        default=ChassisConfig(),
        description="Configuration for the chassis",
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Enable runtime validation of field assignments
        extra="forbid",  # Prevent accidental addition of undefined fields
        frozen=False,  # Allow modification of configuration after initialization
    )
