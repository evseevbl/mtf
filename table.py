import sys
import matplotlib.pyplot as plt
import numpy as np


def smooth_triangle(data, degree):
    triangle = np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed = []
    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point) / sum(triangle))
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def main():
    fname = sys.argv[1]
    fin = open(fname)
    ls = fin.readlines()
    fin.close()
    x = list()
    y1 = list()
    j = 0
    while j < len(ls):
        pair = ls[j].split(';')
        x.append(float(pair[0].strip()))
        y1.append(float(pair[1].strip()))
        j += 1
    y = smooth_triangle(y1, 2)
    l1, = plt.plot(x, y1, 'g--')
    l2, = plt.plot(x, y, 'r-')

    plt.legend([l1, l2], ["mtf", "mtf_smooth"])
    plt.xlabel("frequency, 1/px")
    plt.ylabel("contrast, [0..1]")
    plt.show()


if __name__ == '__main__':
    main()
