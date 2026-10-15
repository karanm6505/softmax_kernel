"""PyTorch baseline: time torch.softmax on the same shapes as ./softmax.

Usage: python3 bench_torch.py <rows> <cols> [iters]
Prints one CSV line in the same format as the CUDA harness.
"""
import sys

import torch


def main() -> None:
    rows, cols = int(sys.argv[1]), int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    x = torch.empty(rows, cols, device="cuda").uniform_(-8, 8)
    for _ in range(3):  # warm-up
        torch.softmax(x, dim=1)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.softmax(x, dim=1)
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))

    times.sort()
    ms = times[len(times) // 2]
    gbps = 2 * rows * cols * 4 / (ms * 1e-3) / 1e9
    print(f"torch,{rows},{cols},{ms:.4f},{gbps:.1f},,")


if __name__ == "__main__":
    main()
