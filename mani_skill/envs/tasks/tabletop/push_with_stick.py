from typing import Any, Dict, Union

import numpy as np
import torch

from transforms3d.euler import euler2quat

import sapien
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply, quaternion_invert


@register_env("PushWithStick-v1", max_episode_steps=100) # TODO: increase env steps
class PushWithStickEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda #Union[Panda, Xmate3Robotiq, Fetch]
    stick_half_sizes = [0.02, 0.30, 0.02] # [x, y, z] - In camera view, x is in front of robot, y is right, z is up
    cube_half_size = 0.02
    moving_cube_thresh = 0.10# 0.055(default), 0.1(loose), 0.15(looser) # stick diagonal len + cube diagonal len = 0.02*sqrt(2) + 0.02*sqrt(2) ~= 0.0567
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.2, 0.6], [0.0, -0.40, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        # janky way of setting robot pose without messing with scene_builder.py
        # self.agent.robot.set_pose(sapien.Pose([-0.615, -0.2, 0]))
        self.table_scene.build()
        self.stick = actors.build_box(
            self.scene, half_sizes=self.stick_half_sizes, color=[0, 0.5, 1, 1], name="stick"
        )
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube"
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx, 
                custom_robot_pose=[0, -1.2195, 0], custom_robot_q=euler2quat(0, 0, np.pi/2).tolist()) # default is [-0.615, 0, 0]
            
            # TODO: make sure stick doesn't spawn on cube, or vice versa
            stick_xyz = torch.zeros((b, 3))
            # TODO: remove later, but make position static for now
            stick_xyz[:, :2] = torch.tensor([0, -0.4]).repeat((b, 1))
            # stick_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # stick_xyz[:, 1] -= 0.6 # Move closer to robot
            stick_xyz[:, 2] = self.stick_half_sizes[0]
            # TODO: remove later, but make position static for now
            stick_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=[-np.pi/4, np.pi/4])
            self.stick.set_pose(Pose.create_from_pq(stick_xyz, stick_qs))

            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            cube_xyz[:, 1] += 0.05 # Move farther from robot
            cube_xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1# [-0.1, 0.1]
            goal_xyz[:, 0] = cube_xyz[:, 0] + (torch.randint(0, 2, (b,))*2 - 1) * (torch.rand((b,))*.1 + 0.1) # put goal x in [-.2, -.1] U [.1, .2] relative to cube
            goal_xyz[:, 1] = cube_xyz[:, 1] + torch.rand(b) * 0.1 # dont put goal behind cube b/c it might be hard for now
            goal_xyz[:, 2] = cube_xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                stick_pose=self.stick.pose.raw_pose,
                tcp_to_stick_pos=self.rel_dist_to_stick(self.agent.tcp.pose, self.stick.pose), # self.stick.pose.p - self.agent.tcp.pose.p
                obj_to_stick_pos=self.rel_dist_to_stick(self.cube.pose, self.stick.pose) # self.stick.pose.p - self.cube.pose.p 
            )
        return obs
    
    def rel_dist_to_stick(self, obj_pose, stick_pose):
        # Move to stick coordinates
        inv_q = quaternion_invert(stick_pose.q)
        displacement_to_stick = quaternion_apply(inv_q, obj_pose.p) - quaternion_apply(inv_q, stick_pose.p)
        # if y value > stick len: subtract stick len
        # if y value < stick len and > -stick len, set to 0
        # if y value < -stick len, add stick len
        assert self.stick_half_sizes[1] == 0.30
        stick_len = self.stick_half_sizes[1]
        displacement_to_stick[torch.abs(displacement_to_stick[:, 1]) <= stick_len, 1] = 0
        displacement_to_stick[displacement_to_stick[:, 1] > stick_len, 1] -= stick_len
        displacement_to_stick[displacement_to_stick[:, 1] < -stick_len, 1] += stick_len
        return displacement_to_stick

    # Returns the distance of object to stick, ignoring placement along stick length
    # y distance is measured from closest end of stick, and 0 within stick length
    # Assumes both poses are in same coordinate frame initially
    def distance_to_stick(self, obj_pose, stick_pose):
        # Move to stick coordinates
        inv_q = quaternion_invert(stick_pose.q)
        displacement_to_stick = quaternion_apply(inv_q, obj_pose.p) - quaternion_apply(inv_q, stick_pose.p)
        # print("obj rel to stick", displacement_to_stick)

        displacement_to_stick[:, 1] = torch.abs(displacement_to_stick[:, 1]) - self.stick_half_sizes[1]
        displacement_to_stick[:, 1] = torch.clamp(displacement_to_stick[:, 1], min=0)
        # print("obj rel to stick fixed", displacement_to_stick)
        to_stick_dist = torch.linalg.norm(
            displacement_to_stick, axis=1
        )
        return to_stick_dist


    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.stick)
        is_pushing = self.distance_to_stick(self.cube.pose, self.stick.pose) <= self.moving_cube_thresh
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "is_pushing": is_pushing
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reward for distance to stick (TODO: any part of stick, not neccessarily the center)
        # print("tcp pose p:", self.agent.tcp.pose.p)
        # print("tcp pose q:", self.agent.tcp.pose.q)
        # print("stick pose p:", self.stick.pose.p)
        # print("stick pose q:", self.stick.pose.q)
        # print("cube pose p:", self.cube.pose.p)
        # print("cube pose q:", self.cube.pose.q)
        tcp_to_stick_dist = self.distance_to_stick(self.agent.tcp.pose, self.stick.pose)
        # print("tcp_to_stick_dist", tcp_to_stick_dist)

        reaching_reward = 1 - torch.tanh(5 * tcp_to_stick_dist)
        # print("reaching_reward", reaching_reward)
        reward = reaching_reward

        # Reward for holding stick
        is_grasped = info["is_grasped"]
        reward += is_grasped

        # Reward for stick being close to cube
        # TODO: implement (should be distance to any part of stick) 
        #                 (maybe just x dimension? but then height z wouldn't be considered. Also if the stick rotates 90 deg that breaks)
        cube_to_stick_dist = self.distance_to_stick(self.cube.pose, self.stick.pose)
        # print("cube_to_stick_dist", cube_to_stick_dist)
        # pushing_reward = 1 - torch.tanh(5 * cube_to_stick_dist)
        pushing_reward = cube_to_stick_dist <= self.moving_cube_thresh #loose: 0.1, looser: 0.15
        # print("pushing_reward", pushing_reward * is_grasped)
        reward += pushing_reward * is_grasped

        # More reward ideas:
        # reward for stick being on correct side of cube (opposite side from goal direction, viewed from top down)
        # velocity of cube moving towards goal reward - similar to position reward tho, i think
        # env ideas: 
        # make goal pos always somewhat left/right of cube, never directly in front
        # make stick shorter and cube/goal closer (think this change should just involve scene init code?)
        # rl idea: more envs for more fast, also maybe increase exploration level?

        # debug cube_to_stick: can robot stop pushing from behind and start pushing from right without losing reward? 
            # need to manually setup scenario in jupyter notebook

        # Reward for getting cube closer to goal
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = (1 - torch.tanh(5 * obj_to_goal_dist))
        reward += place_reward * is_grasped * info["is_pushing"]

        # Reward for not moving when object placed
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
