import argparse 
import os
import json 
import copy
import time

import png
import cv2
import numpy as np
import pyrealsense2 as rs

class IntelConfig:
    """
    Class for loading json config file into depth camera.
    """

    def __init__(self):
        """
        IntelConfig object constructor.

        Args:
            config_path (str): Path to the config file.
        """

        self.DS5_product_ids = [
            "0AD1",
            "0AD2",
            "0AD3",
            "0AD4",
            "0AD5",
            "0AF6",
            "0AFE",
            "0AFF",
            "0B00",
            "0B01",
            "0B03",
            "0B07",
            "0B3A",
            "0B5C",
        ]

    def find_device_that_supports_advanced_mode(self) -> rs.device:
        """
        Searches devices connected to the PC for one compatible with advanced mode.

        Returns:
            rs.device: RealSense device which supports advanced mode.
        """

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if (
                dev.supports(rs.camera_info.product_id)
                and dev.supports(rs.camera_info.name)
                and str(dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids
            ):
                print(
                    "[INFO] Found device that supports advanced mode:",
                    dev.get_info(rs.camera_info.name),
                )
                return dev

        raise Exception(
            "[ERROR] No RealSense camera that supports advanced mode was found"
        )

    def load_config(self, config_path: str):
        """
        Loads json config file into the camera.

        Args:
            config_path (str): Path to the config file.
        """

        # Open camera in advanced mode
        dev = self.find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)

        # Read configuration JSON file as string and print it to console
        # serialized_string = advnc_mode.serialize_json()
        # print(serialized_string)

        # Write configuration file to camera
        with open(config_path) as file:
            data = json.load(file)
        json_string = str(data).replace("'", '"')
        advnc_mode.load_json(json_string)
        print("[INFO] Loaded RealSense camera config from file:", config_path)

class RealSenseCamera:
    """Class for realsesnse camera"""

    def __init__(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        print(devices)
        if len(devices) == 0:
            raise Exception("No device connected, please connect a RealSense device")

        for dev in devices:
            print(dev.get_info(rs.camera_info.name))
            print(dev.get_info(rs.camera_info.serial_number))
            print(dev.get_info(rs.camera_info.product_id))


        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # config_name = "D435_camera_config_defaults.json"
        self.ic = IntelConfig()
        # self.ic.load_config(config_name)
        
        print(self.pipeline)
        print(self.config)

        # dev = self.find
        # Enable streams with the same resolution
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

        # # Enable IMU streams only for D435i
        # self.config.enable_stream(rs.stream.accel)
        # self.config.enable_stream(rs.stream.gyro)
        
        self.align = rs.align(rs.stream.color)

        # Create object for filling missing depth pixels where the sensor was not able to detect depth
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 4)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)

        # Create object for colorizing depth frames
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 1)

        # Start video stream
        self.profile = self.pipeline.start(self.config)

        # Used intrinsics 
        self.color_stream = self.profile.get_stream(rs.stream.color)
        # Color intrinsics are the same for alligned depth frames by design
        self.color_intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
        
        # Raw Depth intrinsics 
        self.depth_stream = self.profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = self.depth_stream.as_video_stream_profile().get_intrinsics()
        self.depth2color_extrinsics = self.depth_stream.as_video_stream_profile().get_extrinsics_to(self.color_stream)


    def save_camera_params(self, path) -> None:

        color_intrinsics = self.color_intrinsics
        color_coeffs = color_intrinsics.coeffs
        color_fx = color_intrinsics.fx
        color_fy = color_intrinsics.fy
        color_ppx = color_intrinsics.ppx
        color_ppy = color_intrinsics.ppy
        color_width = color_intrinsics.width
        color_height = color_intrinsics.height
        color_model = color_intrinsics.model
        K = np.array(
            [
                [color_fx, 0, color_ppx],
                [0, color_fy, color_ppy],
                [0, 0, 1]
            ]
        ).tolist()
    
        colors_params = {
            "Kmx": K,
            "width": color_width,
            "height": color_height,
            "model": str(color_model),
            "coeffs": color_coeffs,
            "fx": color_fx,
            "fy": color_fy,
            "ppx": color_ppx,
            "ppy": color_ppy
        }

        raw_depth_intrinsics = self.depth_intrinsics
        raw_depth_coeffs = raw_depth_intrinsics.coeffs
        raw_depth_fx = raw_depth_intrinsics.fx
        raw_depth_fy = raw_depth_intrinsics.fy
        raw_depth_ppx = raw_depth_intrinsics.ppx
        raw_depth_ppy = raw_depth_intrinsics.ppy
        raw_depth_width = raw_depth_intrinsics.width
        raw_depth_height = raw_depth_intrinsics.height
        raw_depth_model = raw_depth_intrinsics.model
        K_depth = np.array(
            [
                [raw_depth_fx, 0, raw_depth_ppx],
                [0, raw_depth_fy, raw_depth_ppy],
                [0, 0, 1]
            ]
        ).tolist()
        raw_depth_params = {
            "Kmx": K_depth,
            "width": raw_depth_width,
            "height": raw_depth_height,
            "model": str(raw_depth_model),
            "coeffs": raw_depth_coeffs,
            "fx": raw_depth_fx,
            "fy": raw_depth_fy,
            "ppx": raw_depth_ppx,
            "ppy": raw_depth_ppy
        }
        depth2color_extrinsics = self.depth2color_extrinsics
        Rmx_d2c = depth2color_extrinsics.rotation
        tvc_d2c = depth2color_extrinsics.translation
        Tmx_d2c = np.eye(4)
        Tmx_d2c[:3, :3] = np.array(Rmx_d2c).reshape(3, 3)
        Tmx_d2c[:3, 3] = np.array(tvc_d2c).reshape(3)
        Tmx_d2c = Tmx_d2c.tolist()
        depth2color_extrinsics = {
            "Tmx": Tmx_d2c,
            "Rmx": Rmx_d2c,
            "tvc": tvc_d2c
        }
        d = {
            "color": colors_params,
            "raw_depth": raw_depth_params,
            "depth2color_extrinsics": depth2color_extrinsics,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def end_stream(self):
        self.pipeline.stop()

    def get_intrinsics(self):
        return self.color_intrinsics

    def get_aligned_frames(self):
        framset = self.pipeline.wait_for_frames()
        # metadata = framset.get_frame_metadata(rs.frame_metadata_value.actual_exposure)
        timestamp = framset.get_timestamp()
        # print(timestamp, framset.timestamp, framset.frame_timestamp_domain)
        profile = framset.get_profile()
        # print(profile, framset.profile)

        color_frame = framset.get_color_frame()
        depth_frame = framset.get_depth_frame()
        depth_color_frame = self.colorizer.colorize(depth_frame) # Might not be needed but useful for debugging
        
        K_color = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        K_depth = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        T_d2c = depth_frame.get_profile().as_video_stream_profile().get_extrinsics_to(color_frame.get_profile())
        # print("c", K_color)
        # print("d", K_depth, T_d2c)



        # Align the depth frame to color frame
        aligned_frames = self.align.process(framset)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        K_aligned = aligned_depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        T_d2c_aligned = aligned_depth_frame.get_profile().as_video_stream_profile().get_extrinsics_to(color_frame.get_profile())
        # print("a", K_aligned, T_d2c_aligned)

        depth_color_frame = self.colorizer.colorize(aligned_depth_frame)

        frames = color_frame, aligned_depth_frame, depth_color_frame
        
        # # only for D435i
        # gyro = np.assarry(framset[1].as_motion_frame().get_motion_data())
        # accel = np.assarry(framset[0].as_motion_frame().get_motion_data())
        # frames = color_frame, aligned_depth_frame, depth_color_frame, gyro, accel
        
        # NOTE: Maybe add the original depth frame to the return value  
        # time.sleep(1)
        return frames
    

if __name__ == "__main__":
    cam = RealSenseCamera()
    num_img = 0
    cam.save_camera_params("camera_params.json")
    while True:
        color, depth, depth_color = cam.get_aligned_frames()
        cv2.imshow("color", np.asanyarray(color.get_data()))
        cv2.imshow("depth", np.asanyarray(depth_color.get_data()))
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]: # 27 is the ascii code for the escape key
            cam.end_stream()
            break
        elif key == ord('s'):
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())
            print(color_img.shape)  
            print(depth_img.dtype)
            depth_color_img = np.asanyarray(depth_color.get_data())
            cv2.imwrite(f"color_{num_img:06}.png", color_img)
            cv2.imwrite(f"depth_{num_img:06}.png", depth_img)
            cv2.imwrite(f"depthColorized_{num_img:06}.png", depth_color_img)
            num_img += 1
