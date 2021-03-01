#!/usr/bin/env bash
###############################################################################
# READ THIS BEFORE RUNNING THIS SCRIPT
# - You need to install tmux.
#   - Tips: change pane: Ctrl+b →, change window: Ctrl+b n
# - Change default directories as you need
# - If you add more hyper parameters, nest loop or switch the current parameter
# - This scripts will make a lot of directories so please confirm the variables.
# PROCEDURE
#   1. Create a tmux session
#   2. Then run this script
###############################################################################


# compared hyper parameters
reward_lambda=(0.5 1.0 1.5)
num_axis=(2 3 4)
date=$(date '+%m%d%H')
user=hwipsynergy

logdir="/home/${user}/policy/${date}"
mkdir -p ${logdir}

i=1

while [ ${i} -le ${#reward_lambda[@]} ]
do
  tmux select-window -t $((i-1))
  j=1
  while [ ${j} -le ${#num_axis[@]} ]
  do
    tmux select-pane -t $((j-1))
    tmux send-keys "source /home/${user}/Mujoco/venv/bin/activate" C-m # activate python virtual environment
    tmux send-keys "cd /home/${user}/Mujoco/synergy" C-m
    tmux send-keys "python3 -m baselines.her.experiment.train --env GraspBlock-v0 --num_cpu 1 --n_epochs 100 \
                    --logdir ${logdir}/${reward_lambda[i-1]}_${num_axis[j-1]} --num_axis ${num_axis[j-1]} \
                    --reward_lambda ${reward_lambda[i-1]}" C-m

    if [ ${j} -lt ${#num_axis[@]} ]
    then
      tmux split-window -c reward_lambda[${i}] -h
      tmux select-layout even-horizontal
    fi
    j=$((j + 1))
  done

  if [ ${i} -lt ${#reward_lambda[@]} ]
  then
      tmux new-window -c reward_lambda[${i}]
  fi
  i=$((i + 1))
done


