#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief, normalize=True):
    new_belief = np.zeros(15)
    if action == 'F':
        for state in range(0, 15):
            if state == 0:
                new_belief[state] = (0.2 + 0.1) * belief[state] + 0.1 * belief[state+1]
            elif state == 14:
                new_belief[state] = (0.7 + 0.2) * belief[state] + 0.7 * belief[state-1]
            else:
                new_belief[state] = 0.7 * belief[state-1] + 0.2 * belief[state] + 0.1 * belief[state+1]
    else:
        for state in range(0, 15):
            if state == 0:
                new_belief[state] = (0.7 + 0.2) * belief[state] + 0.7 * belief[state+1]
            elif state == 14:
                new_belief[state] = (0.2 + 0.1) * belief[state] + 0.1 * belief[state-1]
            else:
                new_belief[state] = 0.1 * belief[state-1] + 0.2 * belief[state] + 0.7 * belief[state+1]
    if normalize:
        return new_belief / np.sum(new_belief)
    else:
        return new_belief
    
def sensor_model(observation, belief, world):
    p_z_x = np.zeros(15)
    white_map = (world == 1)
    black_map = (world == 0)
    if observation == 1:
        p_z_x[white_map] = 0.7
        p_z_x[black_map] = 0.1
    else:
        p_z_x[white_map] = 0.3
        p_z_x[black_map] = 0.9
    return p_z_x * belief / np.sum(p_z_x * belief)

def recursive_bayes_filter(actions, observations, belief, world):
    belief = sensor_model(observations[0], belief, world)
    for action, observation in zip(actions, observations[1:]):
        belief = motion_model(action, belief, normalize=False)
        belief = sensor_model(observation, belief, world)
    return belief
