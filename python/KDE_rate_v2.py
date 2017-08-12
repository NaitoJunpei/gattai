import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import math

def KDE(spike_times) :
    spike_times = np.array(sorted(spike_times))
    max_value = max(spike_times)
    min_value = min(spike_times)
    T = max_value - min_value

    diff_spike = np.array(sorted(np.diff(spike_times)))
    dt_samp = diff_spike[np.nonzero(diff_spike)][0]
    
    tin = np.linspace(min_value, max_value, min(math.ceil(T / dt_samp), 1.0 * 10 ** 3))
    spike_ab = spike_times[np.nonzero((spike_times >= min(tin)) * (spike_times <= max(tin)))]

    dt = min(diff_spike)

    y_hist = np.histogram(spike_ab, np.append(tin, max_value) - dt / 2)[0]
    L = len(y_hist)
    N = sum(y_hist)
    y_hist = y_hist / (N * dt)

    Wmin = dt
    Wmax = 1 * (max_value - min_value)

    tol = 1e-5
    phi = (math.sqrt(5) + 1) / 2

    a = ilogexp(Wmin)
    b = ilogexp(Wmax)

    c1 = (phi - 1) * a + (2 - phi) * b
    c2 = (2 - phi) * a + (phi - 1) * b

    f1 = CostFunction(y_hist, N, math.exp(c1), dt)[0]
    f2 = CostFunction(y_hist, N, math.exp(c2), dt)[0]

    k = 0
    W = [0] * 20
    C = [0] * 20

    while(abs(b - a) > tol * (abs(c1) + abs(c2)) and k < 20) :
        if(f1 < f2) :
            b = c2
            c2 = c1

            c1 = (phi - 1) * a + (2 - phi) * b

            f2 = f1
            f1, yh1 = CostFunction(y_hist, N, logexp(c1), dt)

            W[k] = logexp(c1)
            C[k] = f1
            optw = logexp(c1)
            y = yh1 / sum(yh1 * dt)
        else :
            a = c1
            c1 = c2

            c2 = (2 - phi) * a + (phi - 1) * b

            f1 = f2
            f2, yh2 = CostFunction(y_hist, N, logexp(c2), dt)

            W[k] = logexp(c2)
            C[k] = f2
            optw = logexp(c2)
            y = yh2 / sum(yh2 * dt)

        k += 1

    nbs = int(1e3)
    yb = np.zeros([nbs, len(tin)])

    for i in range(0, nbs) :
        idx = [math.ceil(np.random.random() * N) for i in range(0, N)]
        xb = spike_ab[idx]
        y_histb = np.histogram(xb, np.append(tin, max_value) - dt / 2)[0] / (dt * N)

        yb_buf = fftkernel(y_histb, optw / dt)
        yb_buf = yb_buf / sum(yb_buf * dt)

        yb[i] = yb_buf          # matlab版では線形補間をしていたが、今回はmatlab版で引数を一つだけとるもののみを実装しているため、ここは省略する

    ybsort = sort(yb)
    y95b = ybsort[math.floor(0.05 * nbs), :]
    y95u = ybsort[math.floor(0.95 * nbs), :]

    y = y * len(spike_times)

    drawKDE(y, tin, y95b, y95u)

    return y95b, y95u
        
def sort(mat) :
    N = len(mat[0])
    for i in range(0, N) :
        mat[:, i] = sorted(mat[:, i])

    return mat

def logexp(x) :
    return math.log(1 + math.exp(x))

def ilogexp(x) :
    return math.log(math.exp(x) - 1)

def CostFunction(y_hist, N, w, dt) :
    yh = fftkernel(y_hist, w / dt) # density

    # formula for density
    C = sum(yh * yh) * dt - 2 * sum(yh * y_hist) * dt + 2 * 1 / (math.sqrt(2 * math.pi) * w * N)
    C *= N * N

    return C, yh

def fftkernel(x, w) :
    # y = fftkernel(x, w)
    # 
    # Function `fftkernel' applies the Gauss kernel smoother to an input signal using FFT algorithm.
    #
    # Input argument
    # x : Sample signal vector
    # w : Kernel bandwidth (the standard deviation) in unit of the sampling resolution of x.
    # Output argument
    # y : Smoothed signal.
    #
    # JULY 7 / 5, 2017 Author Kazuki Nakamura
    # RIKEN Brain Science Insitute
    # http://2000.jukuin.keio.ac.jp/shimazaki
    # 
    # (New correction in version 1)
    # y-axis was multiplied by the number of data, so that
    # y is a time histogram representing the density of spikes.

    L = len(x)
    Lmax = max(1,0, math.floor(L + 3.0 * w))
    n = int(2 ** (nextpow2(Lmax)))

    X = fft.fft(x, n)

    f = (np.array(range(0, n)) + 0.0) / n
    f = np.r_[-f[range(0, int(n / 2) + 1)], f[range(int(n / 2), 1, -1)]]

    K = [math.exp(-0.5 * ((w * 2 * math.pi * f_i) ** 2)) for f_i in f]

    y = fft.ifft(X * K, n)

    y = y[0:L]

    return y

def nextpow2(n) :
    if (n < 0) :
        return 0
    else :
        m = int(math.ceil(math.log2(n)))

        return m
    
def drawKDE(y, t, y95b, y95u) :
    plt.stackplot(t, y)
    plt.ylim(ymin = 0)
    plt.show()
