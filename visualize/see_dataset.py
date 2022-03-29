#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import math
import os

model = load_model_from_path("../assets/hand/hand.xml")
sim = MjSim(model)

dataset_path = "/home/hwipsynergy/Mujoco/synergy/policy/210215/{}"

viewer = MjViewer(sim)

t = 0
pos_num = 0
postures = np.load(dataset_path.format("grasp_dataset_20.npy"))
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
print(actuation_center)
print(postures)

for name in sim.model.actuator_names:
    print(name)

while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    if t > 500:
        t = 0
        pos_num += 1

    sim.data.ctrl[:-1] = actuation_center[:-1] + postures[pos_num] * actuation_range[:-1]
    sim.data.ctrl[:-1] = np.clip(sim.data.ctrl[:-1], ctrlrange[:-1, 0], ctrlrange[:-1, 1])

    # if t > 100 and os.getenv('TESTING') is not None:
    #     break
