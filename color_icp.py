import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time

class ColorIcp:
    def __init__(self, dataset, output_file="tum_trajectory") -> None:
        self.poses = []
        self.dataset = dataset
        self.intrinsics = self.dataset.get_intrinsics()
        self.o3d_pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            480, 640, self.intrinsics
        )
        self.times = []
        self.output_file = output_file

    def rgbd_image(self, image, depth):
        o3d_im = o3d.geometry.Image(image)
        o3d_dpth_im = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_im, o3d_dpth_im, depth_scale=1.0, depth_trunc=4.0
        )
        return rgbd_image

    def get_prediction_model(self):
        if len(self.poses) < 2:
            return np.eye(4)
        return np.linalg.inv(self.poses[-2]) @ self.poses[-1]

    def run(self):
        option = o3d.pipelines.odometry.OdometryOption(
            iteration_number_per_pyramid_level=o3d.utility.IntVector([5, 3, 2])
        )

        image0, depth_img0 = self.dataset[0]
        prev_image = self.rgbd_image(image0, depth_img0)
        for i in tqdm(range(1, len(self.dataset))):
            # motion estimation
            last_pose = self.poses[-1] if self.poses else np.eye(4)
            # need to check the frame of refererence
            initial_position = np.eye(4)
            # read input
            start_time = time.perf_counter_ns()
            image1, depth_img1 = self.dataset[i]
            cur_image = self.rgbd_image(image1, depth_img1)
            # motion compensation
            [
                success_color_term,
                new_pose,
                info,
            ] = o3d.pipelines.odometry.compute_rgbd_odometry(
                cur_image,
                prev_image,
                self.o3d_pinhole_camera_intrinsic,
                initial_position,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
                option,
            )
            prev_image = cur_image
            self.times.append(time.perf_counter_ns() - start_time)
            self.poses.append(new_pose @ last_pose)

    def save_to_tum(self):
        # Open the file for writing
        timestamps = self.dataset.get_frames_timestamps()

        with open(self.output_file+".txt", "w") as file:
            for i in range(len(self.poses)):  # Assuming you have one pose in this example
                translation = self.poses[i][:3, 3]
                rotation_matrix = self.poses[i][:3, :3]
                try:
                    quaternion = Quaternion(
                        matrix=rotation_matrix
                    )  # Default quaternion for identity rotation
                    w, x, y, z = quaternion
                except ValueError as e:
                    r = Rotation.from_matrix(rotation_matrix)
                    quaternion = r.as_quat()
                    w, x, y, z = quaternion

                # Write the pose information to the file
                timestamp = timestamps[i]  # Replace with your actual timestamp
                file.write(
                    f"{timestamp} {translation[0]} {translation[1]} {translation[2]} {x} {y} {z} {w}\n"
                )

        print(f"Trajectory data has been written to {self.output_file}.txt")
