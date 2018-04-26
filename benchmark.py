import argparse
import os
import subprocess
import sys
import torch
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.test:
        if args.internal:
            internal(args)
        else:
            wrapper(args)
    else:
        if args.internal:
            internal(args)
        else:
            datas = []
            params = {
                '--in-channels': [16, 32, 64, 128, 256, 512, 800, 1024, 2048, 4096],
                '--out-channels': [16, 32, 64, 128, 256, 512, 800, 1024, 2048, 4096],
                '--img-size': [4, 8, 16, 32, 64, 128],
                '--kernel-size': [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
                '--stride': [1, 2, 3, 4, 5, 6],
                '--dilation': [1, 2, 3, 4],
                '--groups': [1, 2, 4, 8, 16, 32, 64, 128],
            }
            datas = {}
            for target, values in params.items():
                datas[target] = []
                for test_value in values:
                    data = custom_wrapper(args, target, test_value)
                    datas[target].append(data)
            with open('profile.pkl', 'wb') as f:
                pickle.dump(datas, f)


def wrapper(args):
    """
    Switch profiling on, run this script again, and interpret the results.
    """
    command = "nvprof --quiet --profile-from-start off -o trace.prof -- "
    command += "python3 " + " ".join(sys.argv[:]) + " --internal"

    if os.path.exists("trace.prof"):
        print("Removing old trace file")
        os.remove("trace.prof")

    subprocess.run(command, shell=True)

    # Now read the trace file.
    profile = torch.autograd.profiler.load_nvprof("trace.prof")
    print(profile.key_averages())

    os.remove("trace.prof")


def custom_wrapper(args, changed_param, value):
    command = "nvprof --quiet --profile-from-start off -o trace.prof -- "
    print('Extra args: {} set to {}'.format(changed_param, value))
    command += "python3 " + " ".join(sys.argv[:]) + " --internal " + \
        changed_param + '={}'.format(value)

    if os.path.exists("trace.prof"):
        print("Removing old trace file")
        os.remove("trace.prof")

    subprocess.run(command, shell=True)

    # Now read the trace file.
    profile = torch.autograd.profiler.load_nvprof("trace.prof")
    print(profile.key_averages())
    data = parse_profiled_event(profile.total_average())

    os.remove("trace.prof")
    return data


def parse_profiled_event(event):
    '''
    http://pytorch.org/docs/master/_modules/torch/autograd/profiler.html
    '''
    cpu_time = event.cpu_time
    gpu_time = event.cuda_time
    return [cpu_time, gpu_time]


def internal(args):
    """Profiling is switched on: do convolution."""
    use_gpu = torch.cuda.is_available() and not args.cpu_only

    conv = torch.nn.Conv2d(args.in_channels, args.out_channels,
                           args.kernel_size, padding=args.kernel_size // 2,
                           groups=args.groups, stride=args.stride,
                           dilation=args.dilation)
    data = torch.Tensor(args.batch, args.in_channels, args.img_size,
                        args.img_size)

    if use_gpu:
        torch.backends.cudnn.benchmark = True
        conv = conv.cuda()
        data = data.cuda()

    data = torch.autograd.Variable(data)

    with torch.cuda.profiler.profile():
        warm_up(conv, data)
        with torch.autograd.profiler.emit_nvtx():
            test(conv, data)


def warm_up(model, data):
    """Push the data through the model once before timing anything so CuDNN can
    run its benchmarks and find the best algorithm."""
    test(model, data, iterations=1)


def test(model, data, iterations=100):
    """Apply the model to the data `iterations` times."""
    for i in range(iterations):
        output = model(data)
        # output.backward(gradient=output)


if __name__ == "__main__":
    main()
