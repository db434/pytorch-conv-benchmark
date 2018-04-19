#!/usr/bin/env python3
import pickle
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('ggplot')


def main():
    x_axis = {
        'in-channels': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'out-channels': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'kernel-size': [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
        'stride': [1, 2, 3, 4, 5, 6],
        'dilation': [1, 2, 3, 4],
        'groups': [1, 2, 4, 8, 16, 32, 64, 128],
    }
    with open('profile.pkl', 'rb') as f:
        y_axis = pickle.load(f)
    for key, item in y_axis.items():
        y_axis[key] = np.array(item).T / 100
    components = ['cpu_time', 'gpu_time']
    keys = x_axis.keys()
    first_plot_keys = ['in-channels', 'out-channels', 'kernel-size']
    second_plot_keys = ['stride', 'dilation', 'groups']
    plt.figure(1)
    for i, key in enumerate(first_plot_keys):
        plt.subplot(*[len(first_plot_keys), 1, i+1])
        y_key = '--' + key
        lines = []
        for j, label in enumerate(components):
            line, = plt.plot(
                x_axis[key], y_axis[y_key][j, :], '-o', label=label)
            lines.append(line)
        plt.legend(handles=lines)
        plt.xlabel(key)
        plt.ylabel('time')
    plt.figure(2)
    for i, key in enumerate(second_plot_keys):
        plt.subplot(*[len(second_plot_keys), 1, i+1])
        y_key = '--' + key
        lines = []
        for j, label in enumerate(components):
            line, = plt.plot(
                x_axis[key], y_axis[y_key][j, :], '-o', label=label)
            lines.append(line)
        plt.legend(handles=lines)
        plt.xlabel(key)
        plt.ylabel('time')
    plt.show()


if __name__ == "__main__":
    main()
