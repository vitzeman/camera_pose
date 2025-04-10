# Native packages
import os
import json
import argparse as argp

# Third party packages
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Local packages
from pose_visualizer.camera_pose import CameraPose
from pose_visualizer.pose_visualizer import PoseVisualizer


COLORS_RGB = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 0),
    (255, 255, 255),
]


def visualize_tag_detection(
    image: np.ndarray,
    corners: list,
    ids: list = None,
    color: tuple = (0, 255, 0),
    width: int = 2,
) -> np.ndarray:
    """Visualize the detection of markers by outliying them in the image

    Args:
        image (np.ndarray): Image in BGR format (WxHx3)
        corners (list): list of the detected corners of the markers in the image
        ids (list, optional): Ids of corresponding marker corners. Defaults to None.
        color (tuple, optional): Color of the outline. Defaults to Green.
        width (int, optional): Width of the outline. Defaults to 2.

    Returns:
        np.ndarray: Image with the detected markers outlined
    """
    # print(f"Number of corners: {len(corners)}")
    # print(corners)

    for e, corner in enumerate(corners):
        # corner = corner[0, :, :]
        corner = np.int32(corner)
        # print(corner.shape)
        cv2.polylines(image, corner, True, color, width)
        cv2.circle(image, (corner[0][0][0], corner[0][0][1]), 2 * width + 1, color, -1)
        if ids is not None:
            idx = ids[e]
            cv2.putText(
                image,
                str(idx[0]),
                (corner[0][0][0], corner[0][0][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                width,
            )

    return image


def main():
    ap = argp.ArgumentParser()
    ap.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to the directory containing the required images and calibration data",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--output",
        required=False,
        default="output",
        help="Path to output directory for the results",
        type=str,
    )
    ap.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Show the visualization of the detected markers, and camera poses",
    )
    ap.add_argument(
        "-b",
        "--base",
        required=False,
        default="2",
        help="Base camera id for the coordinate system",
        type=str,
    )
    ap.add_argument(
        "-u",
        "--units",
        required=False,
        default="m",
        help="Units for the camera poses",
        type=str,
        choices=["m", "mm"],
    )

    args = ap.parse_args()
    path = args.path
    visualize = args.visualize
    new_base_id = args.base
    output = args.output
    units = args.units

    print(f"Path: {path}")
    print(f"Visualize: {visualize}")

    board_path = os.path.join(os.getcwd(), "data", "board", "board_mm.json")
    with open(board_path, "r") as f:
        board = json.load(f)

    cameras = {}

    # Detector initialization
    params = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    for file in sorted(os.listdir(path)):
        if not file.endswith(".jpg"):
            continue
        print(f"Processing {file}")
        camera_num = file.strip(".jpg").split("_")[-1]

        params_file = f"camera_{camera_num}.json"
        with open(os.path.join(path, params_file), "r") as f:
            camera_params = json.load(f)

        K = np.array(camera_params["K"])
        dist = np.array(camera_params["dist"])

        # Image undistortion
        image_path = os.path.join(path, f"camera_{camera_num}.jpg")
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, dist, (img_w, img_h), 1, (img_w, img_h)
        )
        img = cv2.undistort(image, K, dist, None, new_K)
        x, y, w, h = roi
        img = img[y : y + h, x : x + w]

        # Detecting the ArUco markers
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejects = detector.detectMarkers(gray)

        image_points = np.array(corners).reshape(-1, 2)
        world_points = np.empty((image_points.shape[0], 3), dtype=np.float32)
        for e, (corner, idx) in enumerate(zip(corners, ids)):
            # Setting up the world points based on the detected corners
            i = idx[0]
            world_points[e * 4 : (e + 1) * 4, :] = np.array(board[str(i)]).reshape(
                -1, 3
            )

        ret, rvec, tvec = cv2.solvePnP(
            world_points,
            image_points,
            new_K,
            np.array([0, 0, 0, 0]),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        Tmx = np.eye(4)
        Tmx[:3, :3] = R.from_rotvec(rvec.flatten()).as_matrix()
        # Tmx[:3, 3] = tvec.flatten() / 1000  # Convert to meters
        Tmx[:3, 3] = tvec.flatten()

        camera = CameraPose()
        camera.Tmx = Tmx
        camera.aspect_ratio = img_w / img_h
        camera.name = f"camera_{camera_num}"
        camera.image_path = image_path
        camera.color = COLORS_RGB[int(camera_num) % len(COLORS_RGB)]
        cameras[camera_num] = camera

        if visualize:
            img = visualize_tag_detection(img, corners, ids)
            img = visualize_tag_detection(img, rejects, color=(0, 0, 255), width=1)
            cv2.imshow("Image", img)
            print("Press any key to continue")
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Set the new camera base
    new_base = cameras[new_base_id].Tmx
    for key, camera in cameras.items():
        T = camera.Tmx
        T = new_base @ np.linalg.inv(T)
        T[:3, 3] = T[:3, 3] * 0.001
        camera.Tmx = T

        T = camera.Tmx * 1000 if units == "mm" else camera.Tmx
        cam_ext = {
            "units": units,
            "T": T.tolist(),
            "R": T[:3, :3].tolist(),
            "t": T[:3, 3].tolist(),
            "q": R.from_matrix(T[:3, :3]).as_quat().tolist(),
            "name": f"camera_{key}",   
        }
        with open(os.path.join(output, f"camera_ext_{key}.json"), "w") as f:
            json.dump(cam_ext, f, indent=2)


    if not visualize:
        quit()

    # Visualize the camera poses in 3D
    pose_visualizer = PoseVisualizer()
    for key, camera in cameras.items():
        # Camera pose visualization
        pose_visualizer.add_camera(camera)

        color = camera.color
        if max(color) > 1:
            color = [val / 255 for val in color]

        image_path = camera.image_path
        pcl_path = image_path.replace(".jpg", ".npy").replace("camera", "pcl")

        print(f"Loading point cloud from {pcl_path}")
        if os.path.exists(pcl_path):
            pcl = np.load(pcl_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl)
            pcd.paint_uniform_color(color)

            cameras_params_path = image_path.replace(".jpg", ".json")
            with open(cameras_params_path, "r") as f:
                cameras_params = json.load(f)
            T_depth2color = np.array(cameras_params["T_depth2color"])

            # Transform the point cloud to the camera frame
            T = camera.Tmx
            T = T @ T_depth2color

            pcd.transform(T)
            pose_visualizer.add_pointcloud(pcd)


    pose_visualizer.draw()


if __name__ == "__main__":
    main()
