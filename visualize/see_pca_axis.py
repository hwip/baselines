#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import math
import os
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/hwipsynergy/Mujoco/synergy")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/gym-grasp/gym_grasp/envs/assets/hand/hand.xml")
sim = MjSim(model)

dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

pca = PCA(n_components=5)
pca.fit(postures)

pc_axis = 1
n = 0
scores = pca.transform(postures)
score_range = [(min(a), max(a)) for a in scores.T]
print(score_range)
trajectory = []
trajectory_len = 500

for i in range(5):
    if i == pc_axis - 1:
        trajectory.append(np.arange(score_range[pc_axis-1][0], score_range[pc_axis-1][1],
                                    (score_range[pc_axis-1][1] - score_range[pc_axis-1][0])/float(trajectory_len))[:500])
    else:
        trajectory.append(np.zeros(trajectory_len))
    print(trajectory[-1].shape)
trajectory = np.array(trajectory).transpose()

print(trajectory)
while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    if t > 5 and n < 499:
        t = 0
        n += 1

    posture = pca.mean_ + pca.inverse_transform(trajectory[n])

    sim.data.ctrl[:-1] = actuation_center[:-1] + posture * actuation_range[:-1]
    sim.data.ctrl[:-1] = np.clip(sim.data.ctrl[:-1], ctrlrange[:-1, 0], ctrlrange[:-1, 1])

