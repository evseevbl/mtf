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
    s_mtf = 0
    s_base = 0

    # fname = sys.argv[1]
    line_types = ['-', '--', '-.', ':']
    colors = ['g', 'b', 'r', 'y']
    lines = list()
    msgs = list()
    for i in range(1, len(sys.argv)):
        s = s_base
        fin = open(sys.argv[i])
        ls = fin.readlines()
        #print(ls)
        fin.close()
        title = ls[0].split(';')
        x = list()
        y1 = list()
        y2 = list()
        j = 1
        while j < len(ls):
            pair = ls[j].split(';')
            #if pair[1].strip() == '0' and pair[2].strip() == '0':
            #    j += 1
            #    continue
            x.append(float(pair[0].strip()))
            y1.append(float(pair[1].strip()))
            y2.append(float(pair[2].strip()))
            j += 1
        if title[0] == 'mtf':
            s = s_mtf
        y1s = smooth_triangle(y1, s)
        y2s = smooth_triangle(y2, s)
        l1, = plt.plot(x, y1s, colors[i - 1] + line_types[0])
        l2, = plt.plot(x, y2s, colors[i - 1] + line_types[1])
        lines.extend([l1, l2])
        msgs.extend(list(map(lambda z: title[0] + '_' + z.strip(), title[1:])))
    plt.legend(lines, msgs)
    # plt.xlabel("frequency, 1/px")
    plt.xlabel("px")
    # plt.ylabel("contrast, [0..1]")
    plt.show()


if __name__ == '__main__':
    main()
