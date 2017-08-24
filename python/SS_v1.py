##########
# SS_v1.pyを実行するには、matplotlib、numpyライブラリが必要です

# 使い方
# SS_v1.pyを、パスが通っているフォルダに置き、
# import SS_v1
# をすると、ファイル内の関数が、SS_v1.(関数名)の形で実行可能になります。

# ユーザーが使用するのはSS関数のみで十分です。
# SS関数は、spike列を引数に取ります。
# spike列の形式は、list、numpy.arrayなどが利用可能です。
# コスト関数が最小になるビン幅を選択し、それに基づいてヒストグラムを描画します。
# ヒストグラム中のビンの数を返します。
########## 



import matplotlib.pyplot as plt
import math
import numpy as np

def SS(spike_times) :
    spike_times = np.array(spike_times)
    max_value   = max(spike_times)
    min_value   = min(spike_times)
    onset       = min_value - 0.001 * (max_value - min_value)
    offset      = max_value + 0.001 * (max_value - min_value)

    for bin_num in range(1, 500) :
        cost = cost_av(spike_times, onset, offset, bin_num, 10)
        if (bin_num == 1 or cost < cost_min):
            # print(bin_num)
            cost_min        = cost
            optimal_bin_num = bin_num

    print(optimal_bin_num)
    drawSS(spike_times, optimal_bin_num)
    return optimal_bin_num

########## 
# cost_f関数
# コスト関数の計算を行います。

# 引数
# spike_times: スパイク列
# start: ヒストグラムに利用する最初の時間
# end: ヒストグラムに利用する最後の時間
# bin_num: ヒストグラムのビンの数

# 返り値
# ヒストグラムのコスト関数の値
########## 

def cost_f(spike_times, start, end, bin_num) :
    bin_width = (end - start) / bin_num
    hist = np.histogram(spike_times, np.linspace(start, end, bin_num + 1))[0]

    av   = np.mean(hist)
    va   = np.mean(hist * hist)

    return ((2.0 * av - (va - av * av)) / (bin_width * bin_width))

########## 
# cost_av関数
# ヒストグラムに利用する最初の時間を変えながらコスト関数を計算し、その平均値を求めます。

# 引数
# spike_times: スパイク列
# onset: スパイク列の記録開始時間
# offset: スパイク列の記録終了時間
# bin_num: ヒストグラムのビンの数
# times: 最初の時間を変える回数

# 返り値
# コスト関数の平均値
##########

def cost_av(spike_times, onset, offset, bin_num, times) :
    temp = 0.0
    bin_width = (offset - onset) / bin_num
    TT = np.hstack([spike_times, spike_times + (offset - onset)])
    for i in range(0, times) :
        start = onset + i * bin_width / times
        end = offset + i * bin_width / times
        temp += cost_f(TT, start, end, bin_num)

    return temp / times

##########
# drawSS関数
# ヒストグラムの描画を行います。

# 引数
# spike_times: スパイク列
# optimal_bin_num: ヒストグラムのビンの数

# 返り値
# なし
########## 

def drawSS(spike_times, optimal_bin_num):
    plt.hist(spike_times, optimal_bin_num)
    plt.yticks([])
    plt.show()
