"""Code for stiching 2 captures by scanner together"""

import os
import json
from pathlib import Path


import cv2


if __name__ == "__main__":
    path2images = "/home/vit/CIIRC/metching_examples/rubic_down/masked_out"
    image_down = path2images + "/000079.png"
    img_down = cv2.imread(image_down)

    path2images = "/home/vit/CIIRC/metching_examples/rubic_up/masked_out"
    image_up = path2images + "/000079.png"
    img_up = cv2.imread(image_up)

    img = cv2.hconcat([img_down, img_up])

    cv2.imshow("image", img)
    cv2.waitKey(0)

    # cv2.


COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (255, 255, 255),
    (0, 0, 0),
]
