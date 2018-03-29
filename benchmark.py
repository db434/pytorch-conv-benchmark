import argparse
import os
import subprocess
import sys
import torch

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

    args = parser.parse_args()
    
    if args.internal:
        internal(args)
    else:
        wrapper(args)
    
def wrapper(args):
    """Switch profiling on, run this script again, and interpret the results."""
    command = "nvprof --quiet --profile-from-start off -o trace.prof -- "
    command += "python3 " + " ".join(sys.argv[:]) + " --internal"
    
    if os.path.exists("trace.prof"):
        print("Removing old trace file", file=stderr)
        os.remove("trace.prof")
    
    subprocess.run(command, shell=True)
    
    # Now read the trace file.
    profile = torch.autograd.profiler.load_nvprof("trace.prof")
    print(profile.key_averages())
    
    os.remove("trace.prof")

def internal(args):
    """Profiling is switched on: do convolution."""
    use_gpu = torch.cuda.is_available() and not args.cpu_only
    
    conv = torch.nn.Conv2d(args.in_channels, args.out_channels,
                           args.kernel_size, padding = args.kernel_size // 2,
                           groups = args.groups, stride = args.stride,
                           dilation = args.dilation)
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
        #output.backward(gradient=output)
    
if __name__ == "__main__":
    main()
