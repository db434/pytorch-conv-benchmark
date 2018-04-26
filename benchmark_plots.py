#!/usr/bin/env python3
import pickle
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import copy



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
        'in-channels': [16, 32, 64, 128, 256, 512, 800, 1024, 2048, 4096],
        'out-channels': [16, 32, 64, 128, 256, 512, 800, 1024, 2048, 4096],
        'kernel-size': [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
        'img-size': [4, 8, 16, 32, 64, 128],
        'stride': [1, 2, 3, 4, 5, 6],
        'dilation': [1, 2, 3, 4],
        'groups': [1, 2, 4, 8, 16, 32, 64, 128],
    }
    # theoretical flops
    TITAN_1080 = 10.6 * (10 ** 12)
    TITAN_1080 = TITAN_1080 / float(2)
    with open('profile.pkl', 'rb') as f:
        y_axis = pickle.load(f)
    for key, item in y_axis.items():
        y_axis[key] = np.array(item).T
    components = ['cpu_time', 'gpu_time']
    keys = x_axis.keys()
    first_plot_keys = ['in-channels', 'out-channels', 'kernel-size', 'img-size']
    second_plot_keys = ['stride', 'dilation', 'groups']
    pp = PdfPages('plot1.pdf')
    plt.figure(1, figsize=(10, 10))
    for i, key in enumerate(first_plot_keys):
        plt.subplot(*[len(first_plot_keys), 1, i+1])
        y_key = '--' + key
        lines = []
        gpu_times = y_axis[y_key][1, :]
        print(gpu_times)
        macs = [compute_macs(configs, (key, value)) for value in x_axis[key]]
        macs_per_second = np.array(macs) / (np.array(gpu_times) * 1e-9)
        line, = plt.plot(
            x_axis[key], macs_per_second, '-o', label='gpu')
        lines.append(line)
        line, = plt.plot(
            x_axis[key],
            len(x_axis[key]) * [TITAN_1080], '-', label='theoretical')
        lines.append(line)
        plt.legend(handles=lines)
        plt.xlabel(key)
        plt.ylabel('Tp (macs/s)')
        plt.tight_layout()
    pp.savefig()
    pp.close()

    pp = PdfPages('plot2.pdf')
    plt.figure(2, figsize=(10, 10))
    for i, key in enumerate(second_plot_keys):
        plt.subplot(*[len(second_plot_keys), 1, i+1])
        y_key = '--' + key
        lines = []
        print(gpu_times)
        gpu_times = y_axis[y_key][1, :]
        macs = [compute_macs(configs, (key, value)) for value in x_axis[key]]
        macs_per_second = np.array(macs) / (np.array(gpu_times) * 1e-9)
        line, = plt.plot(
            x_axis[key], macs_per_second, '-o', label='gpu')
        lines.append(line)
        plt.legend(handles=lines)
        plt.xlabel(key)
        plt.ylabel('Tp (macs/s)')
        plt.tight_layout()
    pp.savefig()
    pp.close()
    import pdb; pdb.set_trace()


def compute_macs(configs, key_value_tuple=None):
    c = copy.deepcopy(configs)
    if key_value_tuple:
        key, value = key_value_tuple
        c[key] = value
    macs = c['batch'] * c['in-channels'] * c['out-channels']
    macs *= (float(c['img-size']) ** 2) * (float(c['kernel-size']) ** 2)
    macs *= (1 / float(c['groups'])) * ((1 / (float(c['stride']) ** 2)))
    return macs


if __name__ == "__main__":
    main()
