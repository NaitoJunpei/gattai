import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def OS(spike_times) :
    spike_times = np.array(spike_times)
    max_value   = max(spike_times)
    min_value   = min(spike_times)
    onset       = min_value - 0.001 * (max_value - min_value)
    offset      = max_value + 0.001 * (max_value - min_value)
    lv          = 0
    ISI         = np.diff(spike_times)

    for i in range(0, len(spike_times) - 2) :
        interval1 = ISI[i]
        interval2 = ISI[i + 1]

        if(interval1 + interval2 != 0) :
            lv += 3 * pow(interval1 - interval2, 2) / (pow(interval1 + interval2, 2) * (len(spike_times) - 2))
        else :
            lv += 3 / (len(spike_times) - 2)

    for bin_num in range(1, 500) :
        # bin_width = (offset - onset) / bin_num
        # count     = [0.0] * bin_num
        # for x in spike_times :
        #     count[int(math.floor((x - onset) / bin_width))] += 1

        # av   = 0.0
        # va   = 0.0
        # w_av = 0.0
        # for x in count :
        #     if (x > 2) :
        #         fano = 2.0 * lv / (3.0 - lv)
        #     else :
        #         fano = 1.0

        #     w_av += fano * x / bin_num
        #     av   += x / bin_num
        #     va   += pow(x, 2) / bin_num

        # cost = (2.0 * w_av - (va - av * av)) / pow(bin_width, 2)
        times = 10
        cost = cost_av(spike_times, onset, offset, lv, bin_num, times)
        
        if (bin_num == 1 or cost < cost_min) :
            cost_min        = cost
            optimal_bin_num = bin_num

    drawOS(spike_times, optimal_bin_num)

def cost_f(spike_times, start, end, lv, bin_num) :
    bin_width = (end - start) / bin_num
    hist = np.histogram(spike_times, np.linspace(start, end, bin_num + 1))[0]

    fano = 2.0 * lv / (3.0 - lv)

    av   = np.mean(hist)
    va   = np.mean(hist * hist)
    w_av = np.mean(hist * fano)
    fano_bin = np.where(hist > 1, fano, 1.0)

    return ((2.0 * np.mean(hist * fano_bin) - (va - av * av)) / (bin_width * bin_width))

def cost_av(spike_times, onset, offset, lv, bin_num, times) :
    temp = 0.0
    bin_width = (offset - onset) / bin_num
    TT = np.hstack([spike_times, spike_times + (offset - onset)])
    for i in range(0, times) :
        start = onset + i * bin_width / times
        end = offset + i * bin_width / times
        temp += cost_f(TT, start, end, lv, bin_num)

    return temp / times

def drawOS(spike_times, optimal_bin_num):
    plt.hist(spike_times, optimal_bin_num)
    plt.yticks([])
    plt.show()
