from src.pose_generator.icosphere_poses import generate_icosphere_views
from src.camera_pose.camera_pose import CameraPose
from src.pose_visualizer import PoseVisualizer


if __name__ == "__main__":
    Tmxs = generate_icosphere_views(3)
    pv = PoseVisualizer()
    for i in range(Tmxs.shape[0]):
        cp = CameraPose()
        cp.Tmx = Tmxs[i,:,:]
        cp.aspect_ratio = 16/9
        cp.units = "m"
        cp.visualization = "both"
        pv.add_camera(cp)
    pv.draw()




