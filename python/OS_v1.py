##########
# OS_v1.pyを実行するには、matplotlib、numpy、pandasライブラリが必要です

# 使い方
# OS_v1.pyを、パスが通っているフォルダに置き、
# import OS_v1
# をすると、ファイル内の関数が、OS_v1.(関数名)の形で実行可能になります。

# ユーザーが使用するのはOS関数のみで十分です。
# OS関数は、スパイク列を引数に取ります。
# スパイク列の形式は、list、numpy.arrayなどが利用可能です。
# コスト関数が最小になるビン幅を選択し、それに基づいてヒストグラムを描画します。
# 値を返しません。
##########

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
        times = 10
        cost = cost_av(spike_times, onset, offset, lv, bin_num, times)
        
        if (bin_num == 1 or cost < cost_min) :
            cost_min        = cost
            optimal_bin_num = bin_num

    drawOS(spike_times, optimal_bin_num)

########## 
# cost_f関数
# コスト関数の計算を行います。

# 引数
# spike_times: スパイク列
# start: ヒストグラムに利用する最初の時間
# end: ヒストグラムに利用する最後の時間
# lv: スパイク列に関するlv値 
# bin_num: ヒストグラムのビンの数

# 返り値
# ヒストグラムのコスト関数の値
########## 


def cost_f(spike_times, start, end, lv, bin_num) :
    bin_width = (end - start) / bin_num
    hist = np.histogram(spike_times, np.linspace(start, end, bin_num + 1))[0]

    fano = 2.0 * lv / (3.0 - lv)

    av   = np.mean(hist)
    va   = np.mean(hist * hist)
    w_av = np.mean(hist * fano)
    fano_bin = np.where(hist > 1, fano, 1.0)

    return ((2.0 * np.mean(hist * fano_bin) - (va - av * av)) / (bin_width * bin_width))

########## 
# cost_av関数
# ヒストグラムに利用する最初の時間を変えながらコスト関数を計算し、その平均値を求めます。

# 引数
# spike_times: スパイク列
# onset: スパイク列の記録開始時間
# offset: スパイク列の記録終了時間
# lv: スパイク列に関するlv値
# bin_num: ヒストグラムのビンの数
# times: 最初の時間を変える回数

# 返り値
# コスト関数の平均値
##########


def cost_av(spike_times, onset, offset, lv, bin_num, times) :
    temp = 0.0
    bin_width = (offset - onset) / bin_num
    TT = np.hstack([spike_times, spike_times + (offset - onset)])
    for i in range(0, times) :
        start = onset + i * bin_width / times
        end = offset + i * bin_width / times
        temp += cost_f(TT, start, end, lv, bin_num)

    return temp / times

##########
# drawOS関数
# ヒストグラムの描画を行います。

# 引数
# spike_times: スパイク列
# optimal_bin_num: ヒストグラムのビンの数

# 返り値
# なし
########## 

def drawOS(spike_times, optimal_bin_num):
    plt.hist(spike_times, optimal_bin_num)
    plt.yticks([])
    plt.show()
