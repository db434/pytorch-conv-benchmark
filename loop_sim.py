import argparse
import math
from functools import reduce


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
    def __init__(self, width=10, depth=10):
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
        throughput = 1 / float(self.factor)
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


def main(args):
    # row, col, fo, fi, i, j
    iterator_bounds = [
        args.out_img, args.out_img, args.out_channels, args.in_channels,
        args.kernel_size, args.kernel_size]
    print('The are {} iterators with the following bounds: {}'.format(
        len(iterator_bounds), iterator_bounds
    ))
    pe = PE()
    total_iters = reduce(lambda x, y: x*y, iterator_bounds)
    for i in range(len(iterator_bounds)):
        unrolling = iterator_bounds[i]
        iters = total_iters / unrolling
        pe.compute(unrolling, iters, i)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    # this is the input row and col size
    parser.add_argument("--in-img", type=int, default=32)
    parser.add_argument("--out-img", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    main(args)
