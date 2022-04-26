import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker

from baselines.her.experiment.pos_database import PosDatabase


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
@click.option('--min_num', type=int, default=100,help='minimum number of success_u whether to run PCA')
@click.option('--num_axis', type=int, default=5,help='number of principal components to calculate the reward function')
@click.option('--reward_lambda', type=float, default=1.,help='a weight for the second term of the reward function')
@click.option('--synergy_type', type=click.Choice(['actuator', 'joint']), default='actuator',
              help='the type of samples calculated in PCA. '
                   '"actuator" uses actuator inputs, "joint" uses joint positions ')

def main(policy_file, seed, n_test_rollouts, render, min_num=10, num_axis=3, reward_lambda=0.7, synergy_type='actuator'):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params['synergy_type'] = synergy_type
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    pos_database = PosDatabase(reward_lambda=0, num_axis=num_axis, init_poslist=[], maxn_pos=200)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts(min_num=min_num, num_axis=num_axis, reward_lambda=reward_lambda,
                                    pos_database=pos_database)

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
