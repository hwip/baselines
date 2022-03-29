from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args


# --------------------------------------------------------------------------------------
from baselines.custom_logger import CustomLoggerObject
clogger = CustomLoggerObject()
clogger.info("MyLogger is working!!")
# --------------------------------------------------------------------------------------

# --Hara work (add by motoda)------
import time
import tensorflow as tf
def tf_pca(data_tensor):
    mean = tf.reduce_mean(data_tensor, axis=0, keepdims=True)
    mean_adj = tf.subtract(data_tensor,mean) #行列から平均を引いているｰ>グラム行列の期待値が分散共分散行列になる（？） # mottoda modify
    # TODO:ちょっと曖昧なので個々の関係はまた調べること
    n_sample = tf.cast(tf.shape(data_tensor)[0]-1, tf.float32) # 不変分散で実装
    # 特異値分解をPCAに戻すときの変換，参考文献のSVDとPCAの関係を参考
    cov = tf.matmul(mean_adj, mean_adj, transpose_a=True) / n_sample # 分散共分散行列を計算して，期待値を計算
    # 特異値分解
    # S:固有値^2, U:固有ベクトルを並べたもの=主成分ベクトル, V=U^T
    S, U, V = tf.linalg.svd(cov)

    # 寄与率の計算
    lambda_vector=tf.divide(S, n_sample) #固有値の計算
    contribution_rate=tf.divide(lambda_vector,tf.reduce_sum(lambda_vector)) #寄与率の計算，固有値/固有値の総和で定義される
    return contribution_rate
def numpy_pca(data):
    mean = np.mean(data, axis=0, keepdims=True)
    data_mean_adj = data - mean
    cov = np.cov(data_mean_adj, rowvar=False)
    U, S, V = np.linalg.svd(cov)
    contribution_rate = S / np.sum(S)
    return contribution_rate
# --------

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self, min_num, num_axis, reward_lambda, pos_database, is_train=True,
                          success_type='Sequence', synergy_type='actuator'):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """

        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # evaluate grasp
        dtime = np.zeros(self.rollout_batch_size)

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        q_vals = []
        fcs = []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net,)
            # clogger.info("compute_Q[{}, {}]: policy_output: {}".format(self.compute_Q, t, policy_output))
            
            if self.compute_Q:
                u, Q, fc = policy_output
                Qs.append(Q)
                q_vals.append(Q.copy())
                if fc.ndim == 1:
                    fc = fc.reshape(1,-1)                            
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                # -- nishimura
                # self.envs[i].set_initial_param(_reward_lambda=reward_lambda, _num_axis=num_axis)
                # --
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])

                    pos = None
                    if synergy_type == 'actuator':
                        pos = u[i][0:20]
                    elif synergy_type == 'joint':
                        # only joints of fingers, except joints of the wrist and the vertical slider.
                        pos = curr_o_new['observation'][5:27]

                    if 'is_success' in info:
                        success[i] = info['is_success']

                        # 継続の判定のため
                        if success[i] > 0 and t > self.T*0.90:  # ステップ数の後半10%になった時に判定を始める
                           dtime[i] += 1
                        else:
                           dtime[i] = 0

                        if success_type == 'Sequence':
                            # 一定時間（dtime），成功判定が継続した場合，把持姿勢を追加
                            if dtime[i] >= 5:
                                pos_database.add_pos(pos)
                                dtime[i] = 0
                        elif success_type == 'Last':
                            # 学習の最後5stepで成功した場合のみver
                            if success[i] > 0 and t > self.T*0.95:
                                pos_database.add_pos(pos)

                        o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            if self.compute_Q:
                fcs.append(fc.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        poss = None
        if synergy_type == 'actuator':
            poss = np.array(acts)[:, :, 0:20]
        elif synergy_type == 'joint':
            poss = np.array(obs)[:, :, 5:27]

        if is_train:
            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals,
                           pos=poss
            )
        else:
            episode = dict(o=obs,
                           u=acts,
                           fc=fcs,
                           g=goals,
                           ag=achieved_goals,
                           q=q_vals,
                           pos=poss
            )
            
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker', variance_ratio=[], num_axis=0, grasp_pose=[]):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        # -- motoda add
        if num_axis > 0 and len(variance_ratio) > 0:
            for i in range(num_axis):
                logs += [('pc_{}'.format(i+1), variance_ratio[i]*100)]
        elif num_axis > 0 and len(variance_ratio) == 0:
            for i in range(num_axis):
                logs += [('pc_{}'.format(i+1), 0.0)]



        logs += [('num_grasp', len(grasp_pose))] 

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
