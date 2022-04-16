import os
import numpy as np
import random

random.seed()

from gym import utils, error
# from gym.envs.robotics import rotations, hand_env
from gym_grasp.envs import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs

from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error

from mujoco_py.generated import const

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def euler2mat(euler):
    r = Rotation.from_euler('xyz', euler, degrees=False)
    return r

# Ensure we get the path separator correct on windows
GRASP_OBJECT_XML = os.path.join('hand', 'grasp_object.xml')


class ManipulateEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(
            self, model_path, target_position, target_rotation,
            target_position_range, reward_type, initial_qpos={},
            randomize_initial_position=True, randomize_initial_rotation=True, randomize_object=True,
            distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
            ignore_z_target_rotation=False,
            target_id=0, num_axis=5, reward_lambda=0.5
    ):
        """Initializes a new Hand manipulation environment.

        Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            randomize_object (boolean)
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
            target_id (int): target id
            num_axis (int): the number of components
            reward_lambda (float) : a weight for the second term of the reward function
        """
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type
        self.ignore_z_target_rotation = ignore_z_target_rotation

        self.object_list = ["box:joint", "apple:joint", "banana:joint", "beerbottle:joint", "book:joint",
                            "needle:joint", "pen:joint", "teacup:joint"]
        self.target_id = target_id
        self.num_axis = num_axis  # the number of components
        self.randomize_object = randomize_object  # random target (boolean)
        self.reward_lambda = reward_lambda  # a weight for the second term of the reward function (float)

        if self.randomize_object == True:
            self.object = self.object_list[random.randrange(0, 8, 1)]  # in case of randomly selected target
        else:
            self.object = self.object_list[self.target_id]  # target

        self.step_n = 0
        self.init_object_qpos = np.array([1, 0.87, 0.4, 1, 0, 0, 0])

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']

        hand_env.HandEnv.__init__(
            self, model_path, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control)
        utils.EzPickle.__init__(self)

    def set_initial_param(self, _reward_lambda, _num_axis, _target_id, _randomize_object):
        self.reward_lambda = _reward_lambda  # a weight for the second term of the reward function (float)
        self.num_axis = _num_axis  # the number of components
        self.target_id = _target_id
        self.randomize_object = _randomize_object

    def _get_achieved_goal(self):
        # Object position and rotation.
        object_qpos = self.sim.data.get_joint_qpos(self.object)
        assert object_qpos.shape == (7,)
        return object_qpos

    # def _randamize_target(self):
    #     self.sim.data.set_joint_qpos("target0:joint", [1, 0.87, 0.4, 1, 0, 0, 0])
    #     # print("##### {} #####".format(self.sim.data.get_joint_qpos("target0:joint")))

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    # GoalEnv methods
    # ----------------------------  

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return success - 1.
        else:
            # d_pos, d_rot = self._goal_distance(achieved_goal, goal)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).

            # Train時のみ処理されるように
            if 'u' not in info:
                return

            c_lambda = info['lambda']
            success = self._is_success(achieved_goal, goal).astype(np.float32)  # 成否（1,0）を取得する
            cpenalty = info["contact_penalty"].T[0]
            gpenalty = info["is_in_grasp_space"].T[0]

            reward = (success - 1.) - c_lambda * (success * info['e']) - cpenalty - gpenalty

            return reward

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        # -- motoda
        if self.randomize_object == True:
            self.object = self.object_list[random.randrange(0, 8, 1)]  # in case of randomly selected target
        else:
            self.object = self.object_list[self.target_id]  # target
        # --
        initial_qpos = self.init_object_qpos
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        self.sim.data.set_joint_qpos("robot0:rollhinge", 1.57) # self.np_random.uniform(0, 3.14))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.initial_qpos = initial_qpos
        self.sim.data.set_joint_qpos(self.object, initial_qpos)
        self.step_n = 0

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            # cube_middle_idx = self.object
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(23))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos(self.object)[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.sim.data.get_joint_qpos(self.object)[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.sim.data.get_joint_qpos(self.object)
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _check_contact(self):
        return 0.1 if np.max(self.sim.data.sensordata[-17:]) > 2.0 else 0.0

    def _get_contact_forces(self):
        return self.sim.data.sensordata[-17:]

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        # object_qvel = self.sim.data.get_joint_qvel(self.object)
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        sensordata = self._get_contact_forces()
        # The self.object_id is an important feature
        # but does only one value in the observation array have a positive effect on RL?

        observation = np.concatenate([robot_qpos, achieved_goal, sensordata])
        # observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal, [self.target_id]])
        # observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal]) # temp

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

    def _get_grasp_center_space(self, radius=0.07):
        pospalm = self.sim.data.site_xpos[self.sim.model.site_name2id("robot0:Tch_palm")]
        rotpalm = Rotation.from_matrix(
            self.sim.data.site_xmat[self.sim.model.site_name2id("robot0:Tch_palm")].reshape([3, 3]))

        pospalm = pospalm + rotpalm.apply([0, 0, 0.05])
        rotgrasp = rotpalm * euler2mat([np.pi / 2, 0, 0])
        posgrasp = pospalm + rotgrasp.apply([0, 0, radius])
        return posgrasp

    def _is_in_grasp_space(self, radius=0.05):
        posgrasp = self._get_grasp_center_space(radius=radius)
        posobject = self.sim.data.site_xpos[self.sim.model.site_name2id("box:center")]
        return mean_squared_error(posgrasp, posobject, squared=False) < 0.05

    def _display_grasp_space(self):
        # show a direction for the grasp
        if self.viewer is not None:

            pospalm = self.sim.data.site_xpos[self.sim.model.site_name2id("robot0:Tch_palm")]
            rotpalm = Rotation.from_matrix(
                self.sim.data.site_xmat[self.sim.model.site_name2id("robot0:Tch_palm")].reshape([3, 3]))

            pospalm = pospalm + rotpalm.apply([0, 0, 0.05])
            rotgrasp = rotpalm * euler2mat([np.pi / 2, 0, 0])

            self.viewer.add_marker(type=const.GEOM_ARROW,
                                   pos=pospalm,
                                   label=" ",
                                   mat=rotgrasp.as_matrix(),
                                   size=(0.005, 0.005, 0.2),
                                   rgba=(1, 0, 0, 0.8),
                                   emission=1)

            posgrasp = pospalm + rotgrasp.apply([0, 0, 0.07])

            self.viewer.add_marker(type=const.GEOM_SPHERE,
                                   pos=posgrasp,
                                   label=" ",
                                   size=(0.07, 0.07, 0.07),
                                   rgba=(0, 1, 0, 0.2),
                                   emission=1)

    def _log_contacts(self):
        for j in range(self.sim.data.ncon):
            contact = self.sim.data.contact[j]
            if self.sim.model.geom_id2name(contact.geom2) == 'object' and self.sim.model.geom_id2name(
                    contact.geom1) is not None:
                print('contact {} dist {}'.format(j, contact.dist))
                print('   contact pos ', contact.pos)
                print('   geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                print('   geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))

                # There's more stuff in the data structure
                # See the mujoco documentation for more info!
                geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[j].geom2]
                print('   Contact force on geom2 body', self.sim.data.cfrc_ext[geom2_body])
                print('   norm', np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
                # Use internal functions to read out mj_contactForce
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, j,
                                                    c_array)  # (f, n) f: 作用する力，n: トルク
                print('   c_array', c_array)
                # A 6D vector specifying the collision forces/torques[3D force + 3D torque] between the given groups. Vector of 0's in case there was no collision.

    def _display_contacts(self):
        for j in range(self.sim.data.ncon):
            contact = self.sim.data.contact[j]
            if self.sim.model.geom_id2name(contact.geom2) == 'object' and self.sim.model.geom_id2name(
                    contact.geom1) is not None:
                # geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[j].geom2]
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, j,
                                                    c_array)  # (f, n) f: 作用する力，n: トルク
                # print(rotforce.as_matrix())
                if self.viewer is not None:
                    rotgeom1 = Rotation.from_matrix(
                        self.sim.data.geom_xmat[contact.geom1].reshape([3, 3])) * euler2mat([np.pi / 2, 0, 0])
                    rotnorm = Rotation.from_matrix(contact.frame.reshape([3, 3]))
                    # rotforce = Rotation.from_euler("xyz", c_array[:3])
                    self.viewer.add_marker(type=const.GEOM_ARROW,
                                           pos=contact.pos,
                                           label=" ",
                                           mat=(rotnorm).as_matrix(),
                                           size=(0.002, 0.002, 1.0 * mean_squared_error(c_array[:3], [0, 0, 0])),
                                           rgba=(1, 0, 0, 0.8),
                                           emission=1)

    def step(self, action):
        self.step_n += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            "contact_penalty": self._check_contact(),
            "is_in_grasp_space": 0.0 if self._is_in_grasp_space() else 0.3
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        if self.step_n < 20:
            self.sim.data.set_joint_qpos(self.object, self.initial_qpos)

        # Options for displaying information
        # self._display_contacts()
        # if self._is_in_grasp_space():
        #     self._display_grasp_space()

        return obs, reward, done, info


class GraspObjectEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type="not_sparse"):
        super(GraspObjectEnv, self).__init__(
            model_path=GRASP_OBJECT_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.025, 0.025), (-0.025, 0.025), (0.2, 0.25)]),
            randomize_initial_position=False, reward_type=reward_type,
            distance_threshold=0.05,
            rotation_threshold=100.0,
            randomize_object=False, target_id=0, num_axis=5, reward_lambda=0.4
        )


'''
Object_list:
    self.object_list = ["box:joint", "apple:joint", "banana:joint", "beerbottle:joint", "book:joint",
                            "needle:joint", "pen:joint", "teacup:joint"]
'''
