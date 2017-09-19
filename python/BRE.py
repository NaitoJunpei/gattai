##########
# BRE.pyを実行するには、matplotlib、numpyライブラリが必要です

# 使い方
# BRE.pyを、パスが通っているフォルダに置き、
# import BRE
# をすると、ファイル内の関数が、BRE.(関数名)の形で実行可能になります。

# ユーザーが使用するのはBRE関数のみで十分です。
# BRE関数は、spike列を引数に取ります。
# spike列の形式は、list、numpy.arrayなどが利用可能です。
# EMアルゴリズムでパラメータを推定し、グラフを描画します。
# 値を返しません。
##########

import matplotlib.pyplot as plt
import numpy as np
import math

def BRE(spike_times) :
    spike_times = np.array(list(spike_times))
    max_value   = max(spike_times)
    min_value   = min(spike_times)

    ISI    = np.diff(spike_times)
    mu     = len(spike_times) / (max_value - min_value)
    beta0  = pow(mu, -3)
    beta   = EMmethod(ISI, beta0)
    kalman = KalmanFilter(ISI, beta)

    drawBRE(spike_times, kalman)

def EMmethod(ISI, beta0) :
    N = len(ISI)
    beta = 0
    beta_new = beta0

    for j in range(0, 100) :
        beta = beta_new
        kalman = KalmanFilter(ISI, beta)

        beta_new = 0
        t0 = 0

        for i in range(0, N - 1) :
            if(ISI[i] > 0) :
                beta_new += (kalman[1][i + 1] + kalman[1][i] - 2 * kalman[2][i]
                             + (kalman[0][i + 1] - kalman[0][i])
                             * (kalman[0][i + 1] - kalman[0][i])) / ISI[i]
            else :
                t0 += 1

        beta_new = (N - t0 - 1) / (2 * beta_new)

    return beta_new

def KalmanFilter(ISI, beta) :
    N = len(ISI)
    IEL = N / sum(ISI)
    IVL = pow(IEL / 3, 2)
    A = IEL - ISI[0] * IVL
    EL = np.empty([2, N])
    VL = np.empty([2, N])

    EL_N = np.empty(N)
    VL_N = np.empty(N)
    COVL_N = np.empty(N)

    EL[0][0] = (A + math.sqrt(A * A + 4 * IVL)) / 2
    VL[0][0] = 1 / (1 / IVL + 1 / pow(EL[0][0], 2))

    # prediction and filtering
    for i in range(0, N - 1) :
        EL[1][i] = EL[0][i]
        VL[1][i] = VL[0][i] + ISI[i] / (2 * beta)

        A = EL[1][i] - ISI[i + 1] * VL[1][i]
        EL[0][i + 1] = (A + math.sqrt(A * A + 4 * VL[1][i])) / 2
        VL[0][i + 1] = 1 / (1 / VL[1][i] + 1 / pow(EL[0][i + 1], 2))

    # smoothing
    EL_N[N - 1] = EL[0][N - 1]
    VL_N[N - 1] = VL[0][N - 1]

    for i in range(0, N - 1) :
        i = N - 2 - i
        H = VL[0][i] / VL[1][i]

        EL_N[i] = EL[0][i] + H * (EL_N[i + 1] - EL[1][i])
        VL_N[i] = VL[0][i] + H * H * (VL_N[i + 1] - VL[1][i])
        COVL_N[i] = H * VL_N[i + 1]

    return [EL_N, VL_N, COVL_N]

def drawBRE(spike_times, kalman) :
    xaxis = []
    yaxis = kalman[0][:]
    for i in range(0, len(spike_times) - 1) :
        xaxis.append((spike_times[i] + spike_times[i + 1]) / 2)

    plt.stackplot(xaxis, yaxis)
    plt.show()
