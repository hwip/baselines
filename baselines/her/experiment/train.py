import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

from subprocess import CalledProcessError

# --------------------------------------------------------------------------------------
from baselines.custom_logger import CustomLoggerObject
clogger = CustomLoggerObject()
clogger.info("MyLogger is working!!")
# --------------------------------------------------------------------------------------


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')
    best_policy_grasp_path = os.path.join(logger.get_dir(), "grasp_dataset_on_best_policy.npy") # motoda
    path_to_grasp_dataset = os.path.join(logger.get_dir(), "grasp_dataset_{}.npy") # motoda

    logger.info("Training...")
    best_success_rate = -1

    # motoda
    all_success_u = [] # motoda
    #if os.path.exists("initial_variance_ratio.npy"):
    #    init_success_u = np.load("initial_variance_ratio.npy")
    #    adll_success_u += init_success_u

    if policy.bc_loss == 1: policy.initDemoBuffer(demo_file) #initialize demo buffer if training with demonstrations
    for epoch in range(n_epochs):
        clogger.info("Start: Epoch {}/{}".format(epoch, n_epochs))
        # train
        rollout_worker.clear_history()
        success_u = [] # motoda
        for _ in range(n_cycles):
            episode, success_tmp = rollout_worker.generate_rollouts() # motoda
            # clogger.info("Episode = {}".format(episode.keys()))
            # for key in episode.keys():
            #     clogger.info(" - {}: {}".format(key, episode[key].shape))
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()
            success_u += success_tmp # motoda

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
            np.save(best_policy_grasp_path, success_u)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)
            # -- motoda added
            grasp_path = path_to_grasp_dataset.format(epoch)
            logger.info('Saving grasp pose: {} grasps. Saving policy to {} ...'.format(len(success_u), grasp_path))
            np.save(grasp_path, success_u)
            # --

        all_success_u += success_u

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
        demo_file, logdir_tf=None, logdir_tf_init=None, override_params={}, save_policies=True, 
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()


    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    clogger.info(policy.sess)
    # Prepare for Saving Network
    clogger.info("logdir_tf: {}".format(logdir_tf))
    if not logdir_tf == None:
        clogger.info("Create tc.Saver()")
        import tensorflow as tf
        saver = tf.train.Saver()

    # -- Motoda---------------------------------
    # Load Learned Parameters
    if logdir_tf_init:
        saver.restore(policy.sess, logdir_tf_init)
    # -------------------------------------------
    

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, demo_file=demo_file)


    # Save Trained Network
    if logdir_tf:
        clogger.info("Save tf.variables to {}".format(logdir_tf))
        clogger.info(policy.sess)
        saver.save(policy.sess, logdir_tf)
        clogger.info("Model was successflly saved [logidr_tf={}]".format(logdir_tf))


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
@click.option('--logdir_tf', type=str, default=None, help='the path to save tf.variables.')
@click.option('--logdir_tf_init', type=str, default=None, help='the path to load tf.variables.')
def main(**kwargs):
    clogger.info("Main Func @her.experiment.train")
    launch(**kwargs)


if __name__ == '__main__':
    main()
