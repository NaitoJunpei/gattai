import matplotlib.pyplot as plt
import math
import numpy as np

def SS(spike_times) :
    max_value = max(spike_times)
    min_value = min(spike_times)
    onset     = min_value - 0.001 * (max_value - min_value)
    offset    = max_value + 0.001 * (max_value - min_value)

    for bin_num in range(1, 500) :
        # bin_width = (offset - onset) / bin_num
        # count     = [0.0] * bin_num
        # for x in spike_times:
        #     count[int(math.floor((x - onset) / bin_width))] += 1

        # av = 0.0
        # va = 0.0
        # for x in count :
        #     av += x / bin_num
        #     va += pow(x, 2.0) / bin_num

        # cost = (2.0 * av - (va - av * av)) / pow(bin_width, 2.0)
        cost = cost_av(spike_times, onset, offset, bin_num, 10)
        if (bin_num == 1 or cost < cost_min):
            # print(bin_num)
            cost_min        = cost
            optimal_bin_num = bin_num

    print(optimal_bin_num)
    drawSS(spike_times, optimal_bin_num)
    return optimal_bin_num

def cost_f(spike_times, start, end, bin_num) :
    bin_width = (end - start) / bin_num
    hist = np.histogram(spike_times, np.linspace(start, end, bin_num + 1))[0]

    av   = np.mean(hist)
    va   = np.mean(hist * hist)

    return ((2.0 * av - (va - av * av)) / (bin_width * bin_width))

def cost_av(spike_times, onset, offset, bin_num, times) :
    temp = 0.0
    bin_width = (offset - onset) / bin_num
    TT = np.hstack([spike_times, spike_times + (offset - onset)])
    for i in range(0, times) :
        start = onset + i * bin_width / times
        end = offset + i * bin_width / times
        temp += cost_f(TT, start, end, bin_num)

    return temp / times


def drawSS(spike_times, optimal_bin_num):
    plt.hist(spike_times, optimal_bin_num)
    plt.yticks([])
    plt.show()
