import sys
import matplotlib.pyplot as plt


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
    l1, = plt.plot(x, y1, 'b-')
    plt.legend([l1], ["mtf"])
    plt.xlabel("frequency, 1/px")
    plt.ylabel("contrast, [0..1]")
    plt.show()


if __name__ == '__main__':
    main()
