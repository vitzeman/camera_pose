import json
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R


def center_board():
    """Center the board in the image"""

    path2data = "data/board/board_mm.json"
    save_path = "data/board/board_centered_mm.json"

    with open(path2data, "r") as f:
        board_original = json.load(f)
    print(f"Loaded board from {path2data}")

    world_size = board_original["paper_size_mm"]
    world_size.append(0)
    world_size = np.array(world_size)

    board_centerd = copy.deepcopy(board_original)

    offset = world_size / 2

    for key, val in board_centerd.items():
        if key in ["tag_size_mm", "paper_size_mm", "num_x", "num_y"]:
            print(f"Skipping {key}")
            continue

        val = np.array(val)
        val -= offset

        val = val.tolist()
        board_centerd[key] = val

    with open(save_path, "w") as f:
        json.dump(board_centerd, f, indent=2)
    print(f"Saved centered board to {save_path}")


def rotate_box():
    path2data = "data/board/board_centered_mm.json"
    save_path = "data/board/board_centered_rotated_mm.json"
    with open(path2data, "r") as f:
        board_original = json.load(f)
    print(f"Loaded board from {path2data}")

    Rmx = R.from_euler("ZYX", [-90.0, 180, 0.0], degrees=True).as_matrix()
    print(f"Rotation matrix: {Rmx}")

    board_rotated = copy.deepcopy(board_original)
    for key, val in board_rotated.items():
        if key in ["tag_size_mm", "paper_size_mm", "num_x", "num_y"]:
            print(f"Skipping {key}")
            continue

        val = np.array(val)
        val = Rmx @ val.T
        val = val.T
        val = val.tolist()
        board_rotated[key] = val
    with open(save_path, "w") as f:
        json.dump(board_rotated, f, indent=2)
    print(f"Saved centered board to {save_path}")


if __name__ == "__main__":
    # center_board()
    rotate_box()
