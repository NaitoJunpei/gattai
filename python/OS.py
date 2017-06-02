import matplotlib.pyplot as plt
import math

def OS(spike_times) :
    max_value = max(spike_times)
    min_value = min(spike_times)
    onset     = min_value - 0.001 * (max_value - min_value)
    offset    = max_value + 0.001 * (max_value - min_value)
    lv        = 0

    for i in range(0, len(spike_times) - 2) :
        interval1 = spike_times[i + 1] - spike_times[i]
        interval2 = spike_times[i + 2] - spike_times[i + 1]

        if(interval1 + interval2 != 0) :
            lv += 3 * pow(interval1 - interval2, 2) / (pow(interval1 + interval2, 2) * (len(spike_times) - 2))
        else :
            lv += 3 / (len(spike_times) - 2)

        for bin_num in range(1, 500) :
            bin_width = (offset - onset) / bin_num
            count     = [0] * bin_num
            for x in spike_times :
                count[math.floor((x - onset) / bin_width)] += 1

            av   = 0
            va   = 0
            w_av = 0
            for x in count :
                if (x > 2) :
                    fano = 2.0 * lv / (3.0 - lv)
                else :
                    fano = 1.0

                w_av += fano * x / bin_num
                av   += x / bin_num
                va   += pow(x, 2) / bin_num

            cost = (2.0 * w_av - (va - av * av)) / pow(bin_width, 2)
            if (bin_num == 1 or cost < cost_min) :
                cost_min        = cost
                optimal_bin_num = bin_num

    print(optimal_bin_num)
    return optimal_bin_num

def DrawOS(spike_times):
    optimal_bin_num = OS(spike_times)
    plt.hist(spike_times, optimal_bin_num)
    plt.show()
