import argparse
import math
from functools import reduce
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


plt.style.use('ggplot')

'''
Example conv loop
for (row=0; row<OutR; row++)
    for (col=0; col<OutC; col++)
        for (fo=0; fo<OutF; fo++)
            for (fi=0; fi<InF; fi++)
                for (i=0; i<K; i++)
                    for (j=0; j<K; j++)
output_fm[fo][row][col]+=weights[fo][fi][i][j]*in_fm[fi][stride*row+i][stride*col+j];



Assumptions
1. Compute throughput: 1 MAC/pe/cycle
2. Memory throughtput:
'''


class PE(object):
    def __init__(self, width=200, depth=1):
        '''
        Args:
            width: The number of parallel PEs.
            depth: How many iterations of execution that one single PE is able
                to handel in one clock cycle. (MACs per PE per clock cycle)
        '''
        self.width = width
        self.depth = depth

    def compute(self, workload_parallelism, num_iterations, target_index):
        latency, real_compute_tp = self.clock_cycles(
            workload_parallelism, num_iterations)
        wr_tp, rd_tp, onchip_size = self.memory(
            workload_parallelism, target_index)
        self.real_compute_tp = real_compute_tp
        self.ideal_compute_tp = workload_parallelism
        self.latency = latency
        self.wr_tp = wr_tp
        self.rd_tp = rd_tp
        self.size = onchip_size

    def clock_cycles(self, wp, ni):
        '''
        how many iterations do the hardware need to
        process all parallel input
        '''
        # num of clock cycles to finish compute
        self.factor = math.ceil(wp / float(self.width))
        throughput = 1 / float(self.factor) * wp
        return (self.factor * ni, throughput)

    def memory(self, wp, target_index):
        '''
        Assumptons/Facts:
            1. A data element is a memory unit
            2. Each conv requires 3 memory transactions (2 reads and 1 write)
            3. Lets associate outpupt_fm, input_fm and weights with different
                iterators
        '''
        # row, col, fo, fi, i, j
        # 0,   1,   2,  3,  4, 5
        if target_index == 2:
            # output channel unrolled, in_fm is reused
            onchip_size = wp * 2 + 1
            read_elemetns = wp + 1
            write_elemetns = wp
        if target_index == 3:
            onchip_size = wp * 2 + 1
            read_elemetns = 2 * wp
            write_elemetns = 1
        else:
            onchip_size = wp * 3
            read_elemetns = wp * 2
            write_elemetns = wp
        write_tp = write_elemetns / float(self.factor)
        read_tp = read_elemetns / float(self.factor)
        return (write_tp, read_tp, onchip_size)


class MetaData(object):
    def __init__(self):
        self.names = ['wr_tps', 'rd_tps', 'compute_tps', 'latencies', 'sizes']
        self.unroll_names = [
            'OutRow', 'OutCol', 'OutChannels', 'InChannels', 'Kernel']
        self.wr_tps = []
        self.rd_tps = []
        self.compute_tps = []
        self.latencies = []
        self.sizes = []

    def add(self, tps, latency, size):
        self.wr_tps.append(tps[0])
        self.rd_tps.append(tps[1])
        self.compute_tps.append(tps[2])
        self.latencies.append(latency)
        self.sizes.append(size)

    def generate_meta(self):
        return np.array(
            [self.wr_tps, self.rd_tps, self.compute_tps, self.latencies,
             self.sizes])


def workload_pe(out_img=32, out_channels=256, in_channels=256, kernel_size=3):
    # row, col, fo, fi, i, j
    iterator_bounds = [
        out_img, out_img, out_channels, in_channels, kernel_size,
        kernel_size]
    # print('The are {} iterators with the following bounds: {}'.format(
    #     len(iterator_bounds), iterator_bounds
    # ))
    pe = PE()
    total_iters = reduce(lambda x, y: x*y, iterator_bounds)
    # meta_data = MetaData()
    meta_data = []
    roofline_data = []
    scatters = []
    # ignore one kernel size dim
    # print('mem tps, ideal compute tp')
    for i in range(len(iterator_bounds)):
        unrolling = iterator_bounds[i]
        iters = total_iters / unrolling
        pe.compute(unrolling, iters, i)
        # tps = [pe.wr_tp, pe.rd_tp, pe.ideal_compute_tp]
        # meta_data.add(tps, pe.size, pe.latency)
        meta_data.append(
            [pe.wr_tp, pe.rd_tp, pe.ideal_compute_tp,  pe.size, pe.latency])
        roofline_data.append([pe.real_compute_tp / float(pe.wr_tp + pe.rd_tp),
            min(pe.real_compute_tp, pe.wr_tp + pe.rd_tp)])
        scatters.append([pe.real_compute_tp, float(pe.wr_tp + pe.rd_tp),
                         pe.latency])
    return (np.array(meta_data), np.array(roofline_data), np.array(scatters))


def main():
    configs = {
        'out_img': 32,
        'out_channels': 128,
        'in_channels': 256,
        'kernel_size': 3,
    }
    test_ranges = {
        'out_img': [16, 32, 64, 128],
        'out_channels': [64, 128, 256, 512, 1024],
        'in_channels': [64, 128, 256, 512, 1024],
        'kernel_size': [1, 2, 4, 8, 16],
    }
    metas = []
    tps = {}
    for key in configs.keys():
        # initialize a empty list for appending
        tps[key] = []
        for value in test_ranges[key]:
            configs[key] = value
            meta_data, roofline_data, scatters = workload_pe(**configs)
            metas.append(meta_data)
            tps[key].append(scatters)
        tps[key] = np.array(tps[key])
        lines = []
        x_axis = []
        x2_axis = []
        y_axis = []
        for i in range(tps[key].shape[1]-1):
            # x_axis.append(tps[key][:, i, 0])
            # x2_axis.append(tps[key][:, i, 2])
            # y_axis.append(tps[key][:, i, 1])
            x_axis = tps[key][:, i, 0]
            y_axis = tps[key][:, i, 1]
            x2_axis = tps[key][:, i, 2]
            # replace contents in a dict to x_axis, y_axis mapping
            lines.append([x_axis, y_axis, x2_axis])
        tps[key] = lines
    # lines = []
    # # minus one on the range because kernels duplicate
        # x_axis = rooflines['out_img'][:, i, 0]
        # y_axis = rooflines['out_img'][:, i, 1]
        # lines.append([x_axis, y_axis])
    # rooflines['out_img'] = lines
    # metas = np.array(metas)
    # # restructure
    # theoreticals = {
    #     'mem_tp': 4, # 10 words per clock cycle, 20 bytes / clock cycle
    #     'compute_tp': 200, # 200 macs per clock cycle
    # }
    for key in configs.keys():
        plot(key, tps[key])


# def plot(theoreticals, x_axis, x_axis_label, metas, figure_id=1):
def plot(key, values, parent_dir='plots/'):
    unroll_names = [
        'OutRow', 'OutCol', 'OutChannels', 'InChannels', 'Kernel']
    colors = ['b', 'g', 'r', 'c', 'y']
    shapes = ['o', '^', 'x', '8', '*']
    pp = PdfPages(parent_dir+key+'.pdf')
    plt.figure(1)
    plt.subplot(211)
    # names = ['wt_tp', 'rd_tp', 'compute_tp', 'on-chip size', 'latency']

    # theoretical line
    # th_mem_tp = theoreticals['mem_tp']
    # th_comp_tp = theoreticals['compute_tp']
    # xlim = 500
    # a couple of points
    # p1 = (0, 0)
    # p2 = (th_comp_tp / float(th_mem_tp), th_comp_tp)
    # p3 = (xlim / float(th_mem_tp), th_comp_tp)
    # points = [p1, p2, p3]
    # th_x_axis, th_y_axis = map(list, zip(*points))

    # line, = plt.plot(
    #         th_x_axis, th_y_axis, '--r', label='theoretical')
    # lines.append(line)
    lines = []

    for i, line in enumerate(values):
        line = plt.scatter(
            line[0], line[1], alpha=0.8, c=colors[i], marker=shapes[i],
            label=unroll_names[i])
        lines.append(line)
    # for j, y_axis in enumerate([wt_tps, rd_tps, compute_tps]):
    #     line, = plt.plot(
    #         x_axis, y_axis, '-o', label=names[j])
    plt.legend(handles=lines)
    plt.xlabel('comp tp (datas / cycle)')
    plt.ylabel('mem tp (datas / cycle)')
    plt.tight_layout()
    ax = plt.subplot(212)
    lines = []
    for i, line in enumerate(values):
        line = plt.scatter(
            line[0], line[2], alpha=0.8, c=colors[i], marker=shapes[i],
            label=unroll_names[i])
        lines.append(line)
    ax.set_xscale("log", nonposx='clip')
    plt.legend(handles=lines)
    plt.ylabel('latencies (cycles)')
    plt.xlabel('comp tp (datas / cycle)')
    # plt.ylabel('mem tp (datas / cycle)')
    plt.tight_layout()
    pp.savefig()


if __name__ == "__main__":
    main()
