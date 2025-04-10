import json

from src.pose_generator.icosphere_poses import generate_icosphere_views
from src.camera_pose.camera_pose import CameraPose
from src.pose_visualizer import PoseVisualizer

from pathlib import Path


def show_icosphere_views(level):
    Tmxs = generate_icosphere_views(level)
    pv = PoseVisualizer()
    for i in range(Tmxs.shape[0]):
        cp = CameraPose()
        cp.Tmx = Tmxs[i, :, :]
        cp.aspect_ratio = 16 / 9
        cp.units = "m"
        cp.visualization = "both"
        pv.add_camera(cp)
    pv.draw()


def from_ngp_transforms():
    path_to_transforms = Path(
        "/home/testbed/Clusters/ClusterHDDs/MLPrague25/Mustard/transform_finetuned.json"
    )
    print(path_to_transforms)
    print(path_to_transforms.exists())
    print(path_to_transforms.parent)

    with open(path_to_transforms, "r") as f:
        transforms = json.load(f)

    fl_x = transforms["fl_x"]
    fl_y = transforms["fl_y"]
    cx = transforms["cx"]
    cy = transforms["cy"]
    w = transforms["w"]
    h = transforms["h"]

    print(fl_x, fl_y, cx, cy, w, h)

def camera_poses():
    """Demo for the camera poses"""
    # TODO: 
    pass


if __name__ == "__main__":
    # show_icosphere_views(3)
    from_ngp_transforms()
