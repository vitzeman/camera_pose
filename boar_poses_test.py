import json
import os

import open3d as o3d

from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Set the default figure size for matplotlib plots
plt.rcParams["figure.figsize"] = [10, 5]  # Values are in inches


def cv2_imshow(bgr: np.ndarray) -> None:
    """Displays the image in jupyter notebook

    Converts the image from BGR to RGB format as cv2 uses BGR formatand matplotlib
    uses RGB format.


    Args:
        bgr (np.ndarray): Image in BGR format [H, W, C]
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()


def visualize_tag_detection(
    image: np.ndarray,
    corners: list,
    ids: list = None,
    color: tuple = (0, 255, 0),
    width: int = 2,
) -> np.ndarray:
    """Visualize the detection of markers by outliying them in the image

    Args:
        image (np.ndarray): Image in BGR format [H, W, C]
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
        corner = np.int32(corner)
        # Tag circumference highlight
        cv2.polylines(image, corner, True, color, width)
        # Indicate the first corner by inputting a circle
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


directry_with_images = Path("MLPrague25_WorkshopDatasetBoard/data/dataset/rgb")
image_files = list(directry_with_images.glob("*.png"))
image_files.sort()

# images = [
#     "data/color_000000.png",
#     "data/color_000001.png",
#     "data/color_000002.png",
#     "data/color_000003.png",
#     "data/color_000004.png",
# ]

images = [str(image) for image in image_files]
for i, image in enumerate(images):
    bgr = cv2.imread(image)
    if bgr is None:
        print(f"Failed to read image {image}")
        continue
    # cv2_imshow(bgr)
    # print(bgr.shape)


# Loading the board data Which contains the marker positions know from the design
board_data_path = "data/board/board_centered_rotated_mm.json"
with open(board_data_path, "r") as f:
    board_data = json.load(f)

# Loading the camera parameters
path2camera_matrix = "MLPrague25_WorkshopDatasetBoard/data/dataset/cam_intrinsics.json"
with open(path2camera_matrix, "r") as f:
    camera_params = json.load(f)

Kmx = np.array(camera_params["K"])
dist_coeffs = np.zeros((5), dtype=np.float32)


params = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
detector = cv2.aruco.ArucoDetector(aruco_dict, params)


camera_poses = []
for i, image in enumerate(images):
    bgr = cv2.imread(image)
    img2show = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Function which returns the image coordinates of the apriltag markers corners
    # and the ids of the markers, also returns the rejected corners
    corners, ids, rejects = detector.detectMarkers(gray)

    num_tags = len(corners)
    print(f"Number of detected tags: {num_tags}")
    corners_image_pts = np.array(corners).reshape(-1, 2)

    # Preallocate the array for the world points for the corners
    corners_world_pts = np.zeros((num_tags * 4, 3), dtype=np.float32)
    # Retrieve corresponding world points for each corner
    for e, tag_id in enumerate(ids):
        i = tag_id[0]
        # Query into the board data
        corners_world_pts[e * 4 : (e + 1) * 4, :] = np.array(
            board_data[str(i)]
        ).reshape(-1, 3)

    print(corners_image_pts.shape)
    print(corners_world_pts.shape)
    ret, Rvec, tvec = ret, rvec, tvec = cv2.solvePnP(
        corners_world_pts,
        corners_image_pts,
        Kmx,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ret:
        print("Failed to solve PnP")
        continue
    # print(f"Rvec: {Rvec}")
    # print(f"tvec: {tvec}")
    Tmx = np.eye(4, dtype=np.float32)
    Tmx[:3, :3] = cv2.Rodrigues(Rvec)[0]
    Tmx[:3, 3] = tvec.reshape(-1)
    # print(f"Tmx: {Tmx}")
    camera_poses.append(Tmx)

    # INFO: SHOWS THE DETECTION OF THE APRILTAG MARKERS AND THE REJECTED CORNERS
    # img2show = visualize_tag_detection(
    #     img2show,
    #     corners,
    #     ids,
    #     color=(0, 255, 0),
    #     width=2,
    # )
    # img2show = visualize_tag_detection(
    #     img2show,
    #     rejects,
    #     color=(0, 0, 255),
    #     width=1,
    # )

    # cv2_imshow(img2show)


def create_board_mesh(img_path: str, board_size: np.ndarray) -> dict:
    """

    Args:
        img_path (str): _description_
        board_size (np.ndarray): _description_

    Returns:
        dict: _description_
    """

    board_image_path = "data/board/board.png"
    board_bgr = cv2.imread(board_image_path)
    board_rgb = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2RGB)
    board_rgb = cv2.flip(board_rgb, 0)
    map_image = o3d.geometry.Image(board_rgb)

    board_size = np.array(board_data["paper_size_mm"])
    half_size = board_size / 2
    print(f"Board size: {board_size}")
    print(f"Half size: {half_size}")
    left_top = np.array([-half_size[0], -half_size[1], 0])
    right_top = np.array([-half_size[0], half_size[1], 0])
    right_bot = np.array([half_size[0], half_size[1], 0])
    left_bot = np.array([half_size[0], -half_size[1], 0])

    vertices = np.array([left_top, right_top, right_bot, left_bot], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32)

    # Map the image to the camera pyramid
    texture_map = o3d.geometry.TriangleMesh()
    texture_map.vertices = o3d.utility.Vector3dVector(vertices)
    texture_map.triangles = o3d.utility.Vector3iVector(triangles)
    texture_map.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    material.base_metallic = 0.0
    material.base_roughness = 1.0
    material.albedo_img = map_image

    return {
        "name": "board",
        "geometry": texture_map,
        "material": material,
    }


def create_camera_pyramid(
    bgr: np.ndarray, cam_K: np.ndarray, Tmx: np.ndarray, name: str = "camera"
) -> None:
    """Based on the camera intrinsics and rgb image creates the camera pyramid

    Adds visualization of the camera pyramid to the scene with the notch
    depicting the top of the image

    Args:
        bgr (np.ndarray): BGR image [HxWx3]
        cam_K (np.ndarray): Camera intrinsics [3x3] matrix
    """
    h, w = bgr.shape[:2]
    camera_lineset = o3d.geometry.LineSet()
    camera_lineset = camera_lineset.create_camera_visualization(
        w, h, cam_K, Tmx, scale=100
    )
    camera_lineset.paint_uniform_color([0.2, 0.2, 0.2])
    camera_points = np.asarray(camera_lineset.points)

    # create top view of the pyramid
    left_top = camera_points[1, :]
    right_top = camera_points[2, :]
    right_bot = camera_points[3, :]
    left_bot = camera_points[4, :]
    height = np.abs(right_bot[1] - right_top[1])
    h = height * 0.33

    # Top Notch

    # NOT CORRECT need to calculate the notch point through vector

    mid_top_point = (right_top - left_top) / 2
    height_vect = (left_top - left_bot) * 0.33

    notch_point = left_top + mid_top_point + height_vect

    # mid_width = (right_top[0] - left_top[0]) / 2
    # notch_point = np.array([left_top[0] + mid_width, left_top[1] - h, left_top[2]])
    notch_points = np.array([left_top, right_top, notch_point])
    top_lineset = o3d.geometry.LineSet()
    top_lineset.points = o3d.utility.Vector3dVector(notch_points)
    top_lineset.lines = o3d.utility.Vector2iVector([[0, 2], [2, 1]])
    top_lineset.paint_uniform_color([0.2, 0.2, 0.2])

    # Add image to the camera pyramid
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 0)
    map_image = o3d.geometry.Image(img)

    vertices = np.array([left_top, right_top, right_bot, left_bot], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32)

    texture_map = o3d.geometry.TriangleMesh()
    texture_map.vertices = o3d.utility.Vector3dVector(vertices)
    texture_map.triangles = o3d.utility.Vector3iVector(triangles)
    texture_map.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    material.base_metallic = 0.0
    material.base_roughness = 1.0
    material.albedo_img = map_image

    camera_img = {
        "name": name,
        "geometry": texture_map,
        "material": material,
    }
    return [camera_lineset, top_lineset, camera_img]


board_mesh = create_board_mesh(
    img_path="data/board/board.png",
    board_size=np.array(board_data["paper_size_mm"]),
)

to_draw = []
to_draw = [board_mesh]
to_draw.append(
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
)  # Origin frame
# sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50)
# sphere.paint_uniform_color([0, 0, 0])  # Black for the origin
# colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
# to_draw.append(sphere)
for e, camera_pose in enumerate(camera_poses):
    if e % 10 != 0:
        continue
    # Create a 4x4 transformation matrix from the camera pose
    Tmx = np.linalg.inv(camera_pose)

    # Create a coordinate frame for the camera pose
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    frame.transform(Tmx)

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
    # sphere.paint_uniform_color(colors[e % len(colors)])  # Color for the camera
    # sphere.transform(Tmx)
    # Visualize the coordinate frame
    to_draw.append(frame)
    # to_draw.append(sphere)

    # Create the camera pyramid
    bgr = cv2.imread(images[e])

    camera_vis = create_camera_pyramid(
        bgr=bgr,
        cam_K=Kmx,
        Tmx=camera_pose,
        name=f"camera_{e}",
    )

    to_draw.extend(camera_vis)


o3d.visualization.draw(to_draw, show_skybox=False)
