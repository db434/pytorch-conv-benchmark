import argparse
import math
from functools import reduce
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np

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
'''


class PE(object):
    def __init__(self, width=100, depth=10):
        '''
        Args:
            width: The number of parallel PEs.
            depth: How many iterations of execution that one single PE is able
                to handel in one clock cycle. (MACs per PE per clock cycle)
        '''
        self.width = width
        self.depth = depth

    def compute(self, workload_parallelism, num_iterations, target_index):
        latency, compute_tp = self.clock_cycles(
            workload_parallelism, num_iterations)
        wr_tp, rd_tp, onchip_size = self.memory(
            workload_parallelism, target_index)
        self.latency = latency
        self.compute_tp = compute_tp
        self.wr_tp = wr_tp
        self.rd_tp = rd_tp
        self.size = onchip_size

    def clock_cycles(self, wp, ni):
        '''
        how many iterations do the hardware need to
        process all parallel input
        '''
        self.factor = math.ceil(wp / float(self.width)) / float(self.depth)
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
    print('The are {} iterators with the following bounds: {}'.format(
        len(iterator_bounds), iterator_bounds
    ))
    pe = PE()
    total_iters = reduce(lambda x, y: x*y, iterator_bounds)
    meta_data = MetaData()
    # ignore one kernel size dim
    for i in range(len(iterator_bounds) - 1):
        unrolling = iterator_bounds[i]
        iters = total_iters / unrolling
        pe.compute(unrolling, iters, i)
        tps = [pe.wr_tp, pe.rd_tp, pe.compute_tp]
        print('unroll factor: ', unrolling)
        print('wr, rd and compute throughputs : {}'.format(tps))
        print('on-chip size {}, latency {}'.format(pe.size, pe.latency))
        meta_data.add(tps, pe.size, pe.latency)
    return meta_data.generate_meta()


def main():
    configs = {
        'out_img': 32,
        'out_channels': 128,
        'in_channels': 128,
        'kernel_size': 3,
    }
    # test out_imgs
    test_range = [2, 4, 8, 16, 32, 128, 256, 512]
    metas = []
    for value in test_range:
        configs['out_img'] = value
        meta_data = workload_pe(**configs)
        metas.append(meta_data)
    # restructure
    plot(test_range, 'img_sizes', metas, 1)
    import pdb; pdb.set_trace()


def plot(x_axis, x_axis_label, metas, figure_id=1):
    plt.figure(1)
    names = ['wt_tp', 'rd_tp', 'compute_tp', 'on-chip size', 'latency']

    plt.figure(figure_id)
    # plot throughputs
    plt.subplot(311)
    lines = []
    for j, y_axis in enumerate([wt_tps, rd_tps, compute_tps]):
        line, = plt.plot(
            x_axis, y_axis, '-o', label=names[j])
        lines.append(line)
    plt.legend(handles=lines)
    plt.xlabel(x_axis_label)
    plt.ylabel('throughputs')

    plt.subplot(312)
    line, = plt.plot(
        x_axis, sizes, '-o', label=names[3])
    plt.legend(handles=[line])
    plt.xlabel(x_axis_label)
    plt.ylabel('on-chip elements')

    plt.subplot(313)
    line, = plt.plot(
        x_axis, latencies, '-o', label=names[4])
    plt.legend(handles=[line])
    plt.xlabel(x_axis_label)
    plt.ylabel('on-chip elements')

    plt.show()


if __name__ == "__main__":
    main()
