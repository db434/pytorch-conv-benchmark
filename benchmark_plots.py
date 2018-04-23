#!/usr/bin/env python3
import pickle
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


plt.style.use('ggplot')


def main():
    configs = {
        'batch': 256,
        'img-size': 32,
        'in-channels': 128,
        'out-channels': 128,
        'kernel-size': 3,
        'stride': 1,
        'dilation': 1,
        'groups': 1,
    }
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
    pp = PdfPages('plot1.pdf')
    plt.figure(1)
    for i, key in enumerate(first_plot_keys):
        plt.subplot(*[len(first_plot_keys), 1, i+1])
        y_key = '--' + key
        lines = []
        gpu_times = y_axis[y_key][1, :]
        macs = [compute_macs(configs, (key, time)) for time in gpu_times]
        line, = plt.plot(
            x_axis[key], macs, '-o', label='gpu')
        lines.append(line)
        plt.legend(handles=lines)
        plt.xlabel(key)
        plt.ylabel('time, macs/s')
        plt.tight_layout()
    pp.savefig()
    pp.close()

    pp = PdfPages('plot2.pdf')
    plt.figure(2)
    for i, key in enumerate(second_plot_keys):
        plt.subplot(*[len(second_plot_keys), 1, i+1])
        y_key = '--' + key
        lines = []
        gpu_times = y_axis[y_key][1, :]
        macs = [compute_macs(configs, (key, time)) for time in gpu_times]
        line, = plt.plot(
            x_axis[key], macs, '-o', label='gpu')
        lines.append(line)
        plt.legend(handles=lines)
        plt.xlabel(key)
        plt.ylabel('time, macs/s')
    pp.savefig()
    pp.close()


def compute_macs(configs, key_value_tuple=None):
    if key_value_tuple:
        key, value = key_value_tuple
        configs[key] = value
    c = configs
    macs = c['batch'] * c['in-channels'] * c['out-channels']
    macs *= (float(c['img-size']) ** 2) * (float(c['kernel-size']) ** 2)
    macs *= (1 / float(c['groups'])) * ((1 / float(c['stride']) ** 2))
    return macs


if __name__ == "__main__":
    main()
