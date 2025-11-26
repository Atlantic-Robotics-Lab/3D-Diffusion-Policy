import sys
import time
import os
import numpy as np
import torch
import pyrealsense2 as rs
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import hydra
from omegaconf import OmegaConf
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R



# Dummy policy import (replace with actual policy loading)
# from diffusion_policy_3d.policy.diffusion_pointcloud_policy import DiffusionPointCloudPolicy
OmegaConf.register_new_resolver("eval", eval, replace=True)

class URDexEnvInference:
    """
    Deployment for UR robot with RealSense camera.
    """
    def __init__(self, ur_ip, obs_horizon=2, action_horizon=8, device="cpu",
                 use_point_cloud=True, use_image=True, img_size=224, num_points=4096):
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.device = torch.device(device)
        self.img_size = img_size
        self.num_points = num_points
        self.ur_ip = ur_ip

        # RealSense setup
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        print("Starting RealSense pipeline...")
        self.pipeline.start(rs_config)
        # try:
        #     while True:
        #         # Wait for a coherent pair of frames: depth and color
        #         frames = self.pipeline.wait_for_frames()
        #         depth_frame = frames.get_depth_frame()
        #         color_frame = frames.get_color_frame()
        #         if not depth_frame or not color_frame:
        #             continue

        #         # Convert images to numpy arrays
        #         depth_image = np.asanyarray(depth_frame.get_data())
        #         color_image = np.asanyarray(color_frame.get_data())
        # finally:
        #     # Stop streaming
        #     self.pipeline.stop()
        #     print("Stopped RealSense pipeline.")
    
        # UR RTDE setup
        self.rtde_c = RTDEControlInterface(ur_ip)
        self.rtde_r = RTDEReceiveInterface(ur_ip)

        # Buffers
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []

    def get_realsense_obs(self, with_rgb=True):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        # After starting the pipeline
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_stream = profile.get_stream(rs.stream.depth)

        # Get intrinsics
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()


        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # meters

        h, w = depth_image.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))  # pixel coordinates

        # Convert depth image to 3D points
        depth_intrinsics = self.depth_intrinsics  # fx, fy, cx, cy
        fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
        cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

        z = depth_image
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy

        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        if with_rgb:
            rgb = color_image.reshape(-1, 3).astype(np.float32) / 255.0
            pc_data = np.concatenate([xyz, rgb], axis=1)  # (H*W,6)
        else:
            pc_data = xyz  # (H*W,3)

        # Optional: downsample to self.num_points
        if pc_data.shape[0] > self.num_points:
            idx = np.random.choice(pc_data.shape[0], self.num_points, replace=False)
            pc_data = pc_data[idx]

        return color_image, depth_image, pc_data

    def get_agent_pos(self):
        # Get current joint positions from UR
        return self.get_ee_pose_quat() #np.array(self.rtde_r.getActualQ())

    def get_ee_pose_quat(self):
        """
        Returns the current UR5e end-effector pose as position + quaternion.
        Output: np.array([x, y, z, qx, qy, qz, qw])
        """
        # Get TCP pose: [x, y, z, rx, ry, rz]  (rotation vector)
        tcp_pose = np.array(self.rtde_r.getActualTCPPose())
        
        # Extract position (meters)
        position = tcp_pose[:3]
        
        # Convert rotation vector (axis-angle) → quaternion
        rotvec = tcp_pose[3:]
        rotation = R.from_rotvec(rotvec)
        quat = rotation.as_quat()  # [x, y, z, w]

        # Combine position + quaternion
        ee_pose_quat = np.concatenate([position, quat])
        
        return ee_pose_quat

    def step(self, action_list):
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            # Send action to UR robot#
            print("Sending home")
            home = [0, -1.57, 1.57, 0, 1.57, 0]
            self.rtde_c.moveJ(home, speed=0.5, acceleration=0.5)
            print("Sending joint target:", act)
            
            #self.rtde_c.moveJ(act, speed=0.5, acceleration=0.5)
            result = self.rtde_c.moveJ(act, speed=0.5, acceleration=0.5)
            print("moveJ result:", result)
            color, depth, cloud = self.get_realsense_obs()
            self.cloud_array.append(cloud)
            self.color_array.append(color)
            self.depth_array.append(depth)
            agent_pos = self.get_agent_pos()
            self.env_qpos_array.append(agent_pos)
        # Build obs_dict
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        return obs_dict

    def reset(self, first_init=True):
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
        color, depth, cloud = self.get_realsense_obs()
        agent_pos = self.get_agent_pos()
        for _ in range(self.obs_horizon):
            self.color_array.append(color)
            self.depth_array.append(depth)
            self.cloud_array.append(cloud)
            self.env_qpos_array.append(agent_pos)
        agent_pos_stack = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        obs_cloud_stack = np.stack([self.cloud_array[-1]]*self.obs_horizon, axis=0)
        obs_img_stack = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_dict = {
            'agent_pos': agent_pos_stack,
        }
        if self.use_point_cloud:
            # obs_cloud_stack = np.stack( [obs_cloud_stack['f0'], obs_cloud_stack['f1'], obs_cloud_stack['f2']], axis=-1).astype(np.float32)

            obs_dict['point_cloud'] = obs_cloud_stack #Test if not unsqueeze
        if self.use_image:
            obs_dict['image'] = obs_img_stack
        return obs_dict

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)

def main(cfg: OmegaConf):

# Use hydra config as in deploy_ori.py
# def run(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)

    # Use policy definition to set variables
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    use_image = getattr(policy, 'use_image', True)
    use_point_cloud = getattr(policy, 'use_point_cloud', True)
    img_size = getattr(policy, 'img_size', 84)
    num_points = getattr(policy, 'num_points', 4096)
    first_init = True
    record_data = True
    roll_out_length = getattr(policy, 'roll_out_length', 100)

    ur_ip = getattr(policy, 'ur_ip', "192.168.1.102")
    obs_horizon = getattr(policy, 'obs_horizon', 2)
    device = getattr(policy, 'device', "cpu")

    env = URDexEnvInference(
        ur_ip=ur_ip,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        device=device,
        use_point_cloud=use_point_cloud,
        use_image=use_image,
        img_size=img_size,
        num_points=num_points
    )
    obs_dict = env.reset(first_init=first_init)
    step_count = 0
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")

    if record_data:
        import h5py
        root_dir = "./ur_deploy_data/"
        save_dir = root_dir + "deploy_dir"
        os.makedirs(save_dir, exist_ok=True)
        record_file_name = f"{save_dir}/demo.h5"
        color_array = np.array(env.color_array)
        depth_array = np.array(env.depth_array)
        cloud_array = np.array(env.cloud_array)
        qpos_array = np.array(env.env_qpos_array)
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("color", data=np.array(color_array))
            f.create_dataset("depth", data=np.array(depth_array))
            f.create_dataset("cloud", data=np.array(cloud_array))
            f.create_dataset("qpos", data=np.array(qpos_array))
        choice = input("whether to rename: y/n")
        if choice == "y":
            renamed = input("file rename:")
            os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
            new_name = record_file_name.replace("demo.h5", renamed+'.h5')
            print(f"save data at step: {roll_out_length} in {new_name}")
        else:
            print(f"save data at step: {roll_out_length} in {record_file_name}")
    # run()

if __name__ == "__main__":
    main()
