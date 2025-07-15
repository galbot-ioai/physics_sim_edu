from typing import Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pathlib import PosixPath
import uuid
from .light import Light
from ..object.object import convert_to_abs_str_path


class LightConfig(BaseModel):
    """Configuration for a light in the scene.

    This class combines the light properties with scene graph information.

    Attributes:
        prim_path (str | PosixPath): The unique primitive path in the scene graph hierarchy. If not provided, defaults to /light/{light_type}/{uuid}
        uuid (str | None): A unique identifier for the entity instance.
        light (RecLight | DistantLight | RectLight | DiskLight | DomeLight | CylinderLight): The light properties and configuration.
    """

    prim_path: Optional[Union[str, PosixPath]] = Field(
        default=None,
        description="The unique primitive path in the scene graph hierarchy. "
        "If not provided, defaults to /light/{light_type}/{uuid}",
    )

    uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_"),
        description="A unique identifier for the entity instance",
    )
    light: Light = Field(..., description="The light properties and configuration")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("prim_path")
    @classmethod
    def validate_prim_path(cls, value: Union[str, PosixPath]) -> str:
        """Convert prim_path to absolute string path.

        Args:
            value: The prim_path to validate

        Returns:
            str: The validated prim_path as an absolute string
        """
        return convert_to_abs_str_path(value)

    @field_validator("uuid", mode="before")
    @classmethod
    def validate_uuid_format(cls, value: str | None) -> str | None:
        """Validate and convert UUID format to use underscores instead of hyphens.

        Args:
            value: UUID string to validate

        Returns:
            str | None: UUID string with underscores, or None if input is None

        Raises:
            ValueError: If UUID format is invalid
        """
        if value is None:
            return None

        # Convert to string if not already
        value = str(value)

        # Replace hyphens with underscores
        value = value.replace("-", "_")

        # Validate UUID format (after replacing hyphens with underscores)
        try:
            # Convert back to standard UUID format for validation
            uuid_obj = uuid.UUID(value.replace("_", "-"))
            # Convert back to underscore format
            return str(uuid_obj).replace("-", "_")
        except ValueError:
            raise ValueError("Invalid UUID format")

    @model_validator(mode="after")
    def set_default_prim_path(self) -> "LightConfig":
        """Set default prim_path if not provided.

        The default prim_path will be /light/{light_type}/{uuid}

        Returns:
            LightConfig: The model instance with default prim_path set
        """
        if (
            self.prim_path is None
            and self.light.type is not None
            and self.uuid is not None
        ):
            self.prim_path = f"/light/{self.light.type}/{self.light.type}_{self.uuid}"
        return self
