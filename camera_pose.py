from dataclasses import dataclass, field
from typing import List, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

M2MM = 1000.0
MM2M = 0.001


@dataclass
class CameraPose:
    """
    Data class to store the camera pose

    Attributes:
    Tmx: np.ndarray: 4x4 camera pose w.r.t. the world frame
    aspect_ratio: float: aspect ratio of the camera image
    units: str: units of the translation (mm or m) # TODO: Not properly implemented
    visualization: str: visualization type (both, pyramid, axes)
    camera_notation: str: camera notation (opencv, opengl)
    image_path: str: path to the camera image
    name: str: camera name
    color: list: color of the camera model (RGB)
    """

    Tmx: np.ndarray = np.eye(4)
    aspect_ratio: float = 1.0
    units: str = "m"
    visualization: str = "both"
    camera_notation: str = "opencv"
    image_path: str = None
    name: str = None
    color: list = field(default_factory=lambda: [0, 0, 0])

    def __setattr__(self, name, value) -> None:
        if name == "Tmx":
            if value.shape != (4, 4):
                raise ValueError("Tmx must be a 4x4 matrix")
        elif name == "aspect_ratio":
            if value <= 0:
                raise ValueError("Aspect ratio must be positive")
        elif name == "units":
            if value not in ["mm", "m"]:
                raise ValueError("Units must be 'mm' or 'm'")
        elif name == "name":
            if type(value) not in [str, type(None)]:
                raise ValueError("Name must be a string")
        elif name == "visualization":
            if value not in ["both", "pyramid", "axes"]:
                raise ValueError("Visualization must be 'both', 'pyramid' or 'axis'")
        elif name == "camera_notation":
            if value not in ["opencv", "opengl"]:
                raise ValueError("Camera notation must be 'opencv' or 'opengl'")
        elif name == "image_path":
            if type(value) not in [str, type(None)]:
                raise ValueError("Image path must be a string")
        elif name == "color":
            if type(value) not in [list, tuple, np.ndarray]:
                raise ValueError("Color must be a list or numpy array")
            if len(value) != 3:
                raise ValueError("Color must be a list of 3 elements")
            if not all([0 <= val <= 255 for val in value]):
                raise ValueError("Color values must be between 0 and 255")
            color = value

        else:
            raise AttributeError(f"Attribute {name} not found")

        super().__setattr__(name, value)

    def __str__(self) -> str:
        angles = R.from_matrix(self.Tmx[:3, :3]).as_euler("zyx", degrees=True)
        angles = [round(angle, 2) for angle in angles]
        string = f"Camera Pose:\n"
        if self.name:
            string += f"\tName: {self.name}\n"
        string += f"\tTranslation: {self.Tmx[:3, 3]} {self.units}\n\tRotation (ZYX): {angles} degrees\n\tAspect Ratio: {self.aspect_ratio}"
        return string


if __name__ == "__main__":
    cP = CameraPose()
    print(cP)
