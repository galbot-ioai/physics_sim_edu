import pytest
import uuid
from pathlib import PosixPath
import numpy as np
from synthnova_config.light import (
    LightConfig,
    Light,
    DistantLight,
    RectLight,
    DiskLight,
    DomeLight,
    CylinderLight,
    SphereLight,
)


def test_light_config_default_values():
    """Test LightConfig with default values."""
    light = DistantLight()
    config = LightConfig(light=light)
    
    assert config.light == light
    assert config.uuid is not None
    assert config.prim_path == f"/light/{light.type}/{config.uuid}"
    assert isinstance(config.uuid, str)
    assert "_" in config.uuid
    assert "-" not in config.uuid


def test_light_config_custom_values():
    """Test LightConfig with custom values."""
    custom_uuid = "123e4567_e89b_12d3_a456_426614174000"  # Valid UUID format
    custom_path = "/custom/path/light"
    light = RectLight()
    
    config = LightConfig(
        prim_path=custom_path,
        uuid=custom_uuid,
        light=light
    )
    
    assert config.prim_path == custom_path
    assert config.uuid == custom_uuid
    assert config.light == light


def test_light_config_prim_path_validation():
    """Test prim_path validation in LightConfig."""
    light = DiskLight()
    
    # Test with string path
    config1 = LightConfig(light=light, prim_path="/test/path")
    assert config1.prim_path == "/test/path"
    
    # Test with PosixPath
    config2 = LightConfig(light=light, prim_path=PosixPath("/test/path"))
    assert config2.prim_path == "/test/path"


def test_light_config_uuid_validation():
    """Test UUID validation in LightConfig."""
    light = DomeLight()
    
    # Test with valid UUID
    valid_uuid = str(uuid.uuid4())
    config1 = LightConfig(light=light, uuid=valid_uuid)
    assert config1.uuid == valid_uuid.replace("-", "_")
    
    # Test with invalid UUID
    with pytest.raises(ValueError):
        LightConfig(light=light, uuid="invalid-uuid")


def test_light_config_with_different_light_types():
    """Test LightConfig with different light types."""
    light_types = [
        DistantLight(),
        RectLight(),
        DiskLight(),
        DomeLight(),
        CylinderLight(),
        SphereLight(),
    ]
    
    for light in light_types:
        config = LightConfig(light=light)
        assert config.light == light
        assert config.prim_path == f"/light/{light.type}/{config.uuid}"


def test_light_config_extra_fields():
    """Test that extra fields are forbidden."""
    light = DistantLight()
    
    with pytest.raises(ValueError):
        LightConfig(light=light, extra_field="test")


def test_light_config_light_validation():
    """Test light property validation."""
    with pytest.raises(ValueError):
        LightConfig(light=None)  # type: ignore


def test_light_config_prim_path_generation():
    """Test automatic prim_path generation."""
    light = CylinderLight()
    config = LightConfig(light=light)
    
    assert config.prim_path.startswith("/light/cylinder/")
    assert config.prim_path.endswith(config.uuid)


def test_light_config_with_custom_light_properties():
    """Test LightConfig with custom light properties."""
    light = RectLight(
        width=2.0,
        height=3.0,
        color=np.array([1.0, 0.5, 0.0]),
        intensity=2.0
    )
    config = LightConfig(light=light)
    
    assert config.light.width == 2.0
    assert config.light.height == 3.0
    assert np.array_equal(config.light.color, np.array([1.0, 0.5, 0.0]))
    assert config.light.intensity == 2.0 