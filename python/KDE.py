import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def KDE(spike_times) :
    spike_times = pd.Series(spike_times)

    optw = search_minimum(spike_times)
    opty = kernel(spike_times, optw)

    drawKDE(opty)

def search_minimum(spike_times) :
    max_value   = max(spike_times)
    min_value   = min(spike_times)
    onset       = min_value - 0.001 * (max_value - min_value)
    offset      = max_value - 0.001 * (max_value - min_value)
    
    def Cost(spike_times, w) :
        A = 0
        for i in range(0, len(spike_times)) :
            var1 = spike_times[i]
            for var2 in spike_times[i + 1:] :
                x = var1 - var2
                if (x < 5 * w) :
                    A += 2 * pow(math.e, (-x * x / (4 * w * w))) - 4 * math.sqrt(2) * pow(math.e, (-x * x / (2 * w * w)))

        return ((len(spike_times) + A) / (w * 2 * math.sqrt(math.pi)))
    
    C_min = np.inf
    for i in range(0, 50):
        W = (max_value - min_value) / (i + 1)
        C = Cost(spike_times, W)

        if (C < C_min) :
            C_min = C
            w = W

    return w

def kernel(spike_times, w) :
    max_value = max(spike_times)
    min_value = min(spike_times)
    K         = 200
    x         = xaxis(K, max_value, min_value)
    y         = [0] * K

    for i in range(0, K) :
        temp = 0
        for spike_time in spike_times :
            diff = x[i] - spike_time
            if (abs(diff) < 5 * w) :
                temp += gauss(diff, w) / len(spike_times)

        y[i] = temp

    return y

def gauss(x, w) :
    return 1 / (w * math.sqrt(2 * math.pi)) * pow(math.e, (-x * x / (2 * w * w)))

def xaxis(K, max_value, min_value) :
    x    = [0] * K
    x[0] = min_value
    for i in range(1, K) :
        x[i] = x[i - 1] + (max_value - min_value) / (K - 1)

    return x


def drawKDE(opty) :
    plt.stackplot(range(0, 200), opty)
    plt.ylim(ymin = 0)
    plt.show()
