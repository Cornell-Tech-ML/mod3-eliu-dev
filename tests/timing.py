import minitorch
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive mode to output file in Google Colab
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Internal timing test for matrix multiplication"""
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y  # noqa: F841, NPY002


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    matrix_sizes = [64, 128, 256, 512, 1024]
    for size in matrix_sizes:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])
    print([times[size]["fast"] for size in matrix_sizes])
    plt.plot(
        [0] + matrix_sizes,
        [0] + [times[size]["fast"] for size in matrix_sizes],
        label="Fast",
        marker="o",
        color="red",
    )
    plt.plot(
        [0] + matrix_sizes,
        [0] + [times[size]["gpu"] for size in matrix_sizes],
        label="GPU",
        marker="x",
        color="blue",
    )
    plt.legend()
    plt.savefig("performance.png")

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
