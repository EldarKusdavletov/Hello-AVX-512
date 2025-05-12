import csv
import math

import matplotlib.pyplot as plt


def y_func(x):
    return float(math.log2(float(x)))


file = "misc/timing.csv"

scalar_times = []
unrolled_times = []
avx2_times = []

with open(file, 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

    if len(lines) >= 3:
        scalar_times = [y_func(x) for x in lines[0] if x.strip()]
        unrolled_times = [y_func(x) for x in lines[1] if x.strip()]
        avx2_times = [y_func(x) for x in lines[2] if x.strip()]

runs = list(range(1, len(avx2_times) + 1))

plt.figure(figsize=(10, 6))
plt.plot(runs, scalar_times, marker='^', label='Scalar')
plt.plot(runs, unrolled_times, marker='s', label='Unrolled')
plt.plot(runs, avx2_times, marker='o', label='AVX2')

plt.xlabel('Run Number')
plt.ylabel('Log(Time (seconds))')
plt.title(f'Performance Comparison Over {len(avx2_times)} Runs')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("misc/img.png")
