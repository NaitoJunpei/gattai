##########
# HMM.pyを実行するには、matplotlib、numpy、pandasライブラリが必要です

# 使い方
# HMM.pyを、パスが通っているフォルダに置き、
# import HMM
# をすると、ファイル内の関数が、HMM.(関数名)の形で実行可能になります。

# ユーザーが使用するのはHMM関数のみで十分です。
# HMM関数は、spike列を引数に取ります。
# spike列の形式は、list、numpy.arrayなどが利用可能です。
# 隠れマルコフモデルを使ってパラメータを推測し、グラフを描画します。
# 値を返しません。
##########

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time

def HMM(spike_times) :
    start = time.time()
    spike_times = np.array(list(spike_times))
    max_value   = max(spike_times)
    min_value   = min(spike_times)
    onset       = min_value - 0.001 * (max_value - min_value)
    offset      = max_value + 0.001 * (max_value - min_value)
    bin_width   = (offset - onset) / len(spike_times) * 5

    rate_hmm = get_hmm_ratefunc(spike_times, bin_width, max_value, min_value)
    end = time.time()
    print(end - start)

    drawHMM(spike_times, rate_hmm)

    return rate_hmm

####
# 隠れマルコフモデルで推定した値の描画を行います。

# 引数
# spike_times: スパイク列
# rate_hmm: 隠れマルコフモデルで推定した値

# 返り値
# なし
####

def drawHMM(spike_times, rate_hmm) :
    xaxis = [rate_hmm[0, 0]]
    yaxis = [rate_hmm[0, 1]]
    tempx_old = tempx = rate_hmm[0, 0]
    tempy_old = tempy = rate_hmm[0, 1]
    for i in range(0, len(rate_hmm)) :
        tempx, tempy = rate_hmm[i]
        if (tempy != tempy_old) :
            mid = (tempx + tempx_old) / 2
            xaxis.append(mid)
            xaxis.append(mid)
            yaxis.append(tempy_old)
            yaxis.append(tempy)
        tempx_old = tempx
        tempy_old = tempy

    xaxis.append(rate_hmm[-1, 0])
    yaxis.append(rate_hmm[-1, 1])
    plt.stackplot(xaxis, yaxis)
    plt.xlim(xmin = min(xaxis), xmax = max(xaxis))
    plt.ylim(ymin = 0)
    plt.show()

####
# get_hmm_ratefunc関数
# EMアルゴリズムを利用し、パラメータを推定しながらスパイクレートを推定します
# 推定されるパラメータには、行列A、ベクトルpi、ベクトルlambda、行列Gamma、ベクトルXi、行列Xiがあります

# 引数
# spike_times: スパイク列
# bin_width: ビン幅 オンセットの時間からオフセットまでの時間をスパイク数で割った値の5倍に設定されています
# max_value: 最後のスパイクの時間
# min_value: 最初のスパイクの時間

# 返り値
# 推定したスパイクレート
####
    
def get_hmm_ratefunc(spike_times, bin_width, max_value, min_value) :
    EMloop_num = 5000
    
    mat_A      = np.array([[0.999, 0.001], [0.001, 0.999]])
    vec_pi     = np.array([0.5, 0.5])
    mean_rate  = len(spike_times) / (max_value - min_value)

    vec_lambda = np.empty(2)

    vec_lambda[0] = (mean_rate * 0.75) * bin_width
    vec_lambda[1] = (mean_rate * 1.25) * bin_width

    vec_spkt = np.array([spike - min_value for spike in spike_times])

    vec_Xi = get_vec_Xi(vec_spkt, bin_width)

    mat_Gamma, mat_Xi = HMM_E_step(vec_Xi, mat_A, vec_lambda, vec_pi)

    mat_A_old      = mat_A.copy()
    vec_pi_old     = vec_pi.copy()
    vec_lambda_old = vec_lambda.copy()
    loop = 0
    flag = 0
    while(loop <= EMloop_num and flag == 0) :
        vec_pi_new, vec_lambda_new, mat_A_new = HMM_M_step(vec_Xi, mat_A, vec_lambda, vec_pi, mat_Gamma, mat_Xi)

        vec_pi     = vec_pi_new.copy()
        vec_lambda = vec_lambda_new.copy()
        mat_A      = mat_A_new.copy()

        sum_check = 0.0
        num_state = len(vec_pi)

        sum_check += sum(abs(vec_pi_old - vec_pi))
        
        sum_check += sum(abs(vec_lambda_old - vec_lambda))

        sum_check += sum(sum(abs(mat_A_old - mat_A)))

        if (sum_check / (1.0 * num_state * (num_state + 2)) < 1.0e-7) :
            flag = 1
            
        mat_A_old      = mat_A.copy()
        vec_pi_old     = vec_pi.copy()
        vec_lambda_old = vec_lambda.copy()
                
        E_res     = HMM_E_step(vec_Xi, mat_A, vec_lambda, vec_pi)
        mat_Gamma = E_res[0]
        mat_Xi    = E_res[1]

        loop += 1

    vec_hidden = HMM_Viterbi(vec_Xi, mat_A, vec_lambda, vec_pi)
    rate_func  = np.empty([len(vec_Xi), 2])

    c_time = 0.0
    print(loop)

    for n in range(0, len(vec_Xi)) :
        state_id        = vec_hidden[n]
        rate_func[n][0] = round(c_time * 100) / 100.0
        rate_func[n][1] = round(vec_lambda[int(state_id)] * 100) / (bin_width * 100.0)

        c_time += bin_width

    return rate_func


def get_vec_Xi(vec_spkt, bin_width) :
    spkt_dura = vec_spkt[len(vec_spkt) - 1]
    bin_num   = int(math.ceil(spkt_dura / bin_width))
    vec_Xi    = np.zeros(bin_num)

    for x in vec_spkt:
        bin_id = int(math.floor(x / bin_width))
        if (bin_id < bin_num) :
            vec_Xi[bin_id] += 1

    return vec_Xi


####
# HMM_E_step関数
# 隠れマルコフモデルの期待値を計算するステップです。

# 引数
# vec_Xi: ベクトルXi、numpy arrayクラスで表現します。
# mat_A: 行列A、numpy arrayクラスで表現します。
# vec_lambda: ベクトルラムダ、numpy arrayクラスで表現します。
# vec_pi: ベクトルパイ、numpy arrayクラスで表現します。

# 返り値
# mat_Gamma: 行列Gamma、numpy arrayクラスで表現します。
# mat_Xi: 行列Xi、numpy arrayクラスで表現します。
####

def HMM_E_step(vec_Xi, mat_A, vec_lambda, vec_pi) :
    mat_emission      = get_mat_emission(vec_Xi, vec_lambda)
    vec_C, mat_alpha  = get_alpha_C(mat_A, vec_pi, mat_emission)
    mat_beta          = get_beta(mat_A, vec_pi, mat_emission, vec_C)
    mat_Gamma, mat_Xi = get_Gamma_Xi(mat_A, mat_emission, mat_alpha, mat_beta, vec_C)

    res = [mat_Gamma, mat_Xi]

    return res

def get_mat_emission(vec_Xi, vec_lambda) :
    mat_emission = np.array([[pow(x, y) * pow(math.e, -1.0 * x) / math.factorial(y) for x in vec_lambda] for y in vec_Xi])

    return mat_emission

def get_alpha_C(mat_A, vec_pi, mat_emission) :
    num_of_states = len(vec_pi)
    num_of_obs    = len(mat_emission)

    alpha_0 = np.array([mat_emission[0][i] * vec_pi[i] for i in range(0, num_of_states)])
    C_0     = sum(alpha_0)

    vec_C_buf     = np.empty(num_of_obs)
    vec_C_buf[0]  = (C_0)
    alpha_0       = alpha_0 / C_0
    mat_alpha_buf = np.zeros([num_of_obs, num_of_states])
    mat_alpha_buf[0] = alpha_0.copy()

    for n in range(1, num_of_obs) :
        row_vec_alpha_buf = mat_alpha_buf[n - 1]
        alpha_n = np.empty([num_of_states])
        J = row_vec_alpha_buf.dot(mat_A)
        alpha_n = mat_emission[n] * J

        C_n = sum(alpha_n)

        vec_C_buf[n] = np.array([C_n])
        alpha_n = alpha_n / C_n

        mat_alpha_buf[n] = alpha_n.copy()

    res = [vec_C_buf, mat_alpha_buf]

    return res


def get_beta(mat_A, vec_pi, mat_emission, vec_C) :
    num_of_states = len(vec_pi)
    num_of_obs    = len(mat_emission)

    # 初期化
    mat_beta_buf = np.zeros([num_of_obs, num_of_states])

    mat_beta_buf[num_of_obs - 1] = 1.0

    for m in range(1, num_of_obs) :
        n               = num_of_obs - 1 - m
        mat_beta_buf_n1 = mat_beta_buf[n + 1]
        mat_emission_n1 = mat_emission[n + 1]
        vec_C_n1        = vec_C[n + 1]
        
        sum_j = mat_A.dot(mat_beta_buf_n1 * mat_emission_n1)
        mat_beta_buf[n] = sum_j / vec_C_n1


    return mat_beta_buf

def get_Gamma_Xi(mat_A, mat_emission, mat_alpha, mat_beta, vec_C) :
    num_of_states = len(mat_emission[0])
    num_of_obs    = len(mat_emission)

    mat_Gamma_buf = np.zeros([num_of_obs, num_of_states])

    mat_Gamma_buf = mat_alpha * mat_beta


    mat_Xi_buf = np.zeros([num_of_obs - 1, num_of_states, num_of_states])
    for m in range(0, num_of_obs - 1) :
        mat_Xi_buf_m    = mat_Xi_buf[m]
        mat_alpha_m     = mat_alpha[m]
        mat_emission_m1 = mat_emission[m + 1]
        mat_beta_m1     = mat_beta[m + 1]
        vec_C_m1        = vec_C[m + 1]

        for i in range(0, num_of_states) :
            mat_Xi_buf_m_i = mat_Xi_buf_m[i]
            mat_alpha_m_i = mat_alpha_m[i]
            mat_A_i = mat_A[i]
            for j in range(0, num_of_states) :
                mat_Xi_buf_m_i[j] = (mat_alpha_m_i * mat_emission_m1[j] * mat_A_i[j] * mat_beta_m1[j]) / vec_C_m1

    res = [mat_Gamma_buf, mat_Xi_buf]

    return res

####
# HMM_M_step関数
# 隠れマルコフモデルの、尤度最大化ステップです。
# ベクトルpi、ベクトルlambda、行列Aの値を更新します。

# 引数
# vec_Xi: ベクトルXi、numpy arrayクラスで表現します。
# mat_A: 行列A、numpy arrayクラスで表現します。
# vec_lambda: ベクトルラムダ、numpy arrayクラスで表現します。
# vec_pi: ベクトルパイ、numpy arrayクラスで表現します。
# mat_Gamma: 行列Gamma、numpy arrayクラスで表現します。
# mat_Xi: 行列Xi、numpy arrayクラスで表現します。

# 返り値
# vec_pi_new: ベクトルパイ、numpy arrayクラスで表現します。
# vec_lambda_new: ベクトルラムダ、numpy arrayクラスで表現します。
# mat_A_new: 行列A、numpy arrayクラスで表現します。

####

def HMM_M_step(vec_Xi, mat_A, vec_lambda, vec_pi, mat_Gamma, mat_Xi) :
    num_of_states = len(mat_A)
    num_of_obs    = len(vec_Xi)

    pi_denom = 0.0
 
    pi_denom += sum(mat_Gamma[0])

    vec_pi_new = mat_Gamma[0] / pi_denom


    # maxmize wrt lambda vector
    vec_lambda_new = np.empty(num_of_states)
    for k in range(0, num_of_states) :
        lambda_denom = 0.0
        lambda_nume  = 0.0
        for n in range(0, num_of_obs) :
            lambda_denom += mat_Gamma[n][k]
            lambda_nume  += mat_Gamma[n][k] * vec_Xi[n]

        if(lambda_denom == 0.0) :
            vec_lambda_new[k] = 0.0
        else :
            vec_lambda_new[k] = lambda_nume / lambda_denom

    # maxmize wrt A matrix
    mat_A_new = np.zeros([num_of_states, num_of_states])
    for j in range(0, num_of_states) :
        A_denome = 0.0
        for n in range(0, num_of_obs - 1) :
            for mat_Xi_n_j_l in mat_Xi[n][j] :
                A_denome += mat_Xi_n_j_l

        for k in range(0, num_of_states) :
            A_nume = 0.0
            for mat_Xi_n in mat_Xi :
                A_nume += mat_Xi_n[j][k]
            if(A_denome == 0.0) :
                mat_A_new[j][k] = 0.0
            else :
                mat_A_new[j][k] = A_nume / A_denome

    res = [vec_pi_new, vec_lambda_new, mat_A_new]
    return res


####
# HMM_Viterbi関数
# Viterbiアルゴリズムによる探索を行います。

# 引数
# vec_Xi: ベクトルXi、numpy arrayクラスで表現します。
# mat_A: 行列A、numpy arrayクラスで表現します。
# vec_lambda: ベクトルラムダ、numpy arrayクラスで表現します。
# vec_pi: ベクトルパイ、numpy arrayクラスで表現します。

# 返り値
# vec_hs_seq: 最適なvec_lambdaのインデックスを返します。
####
def HMM_Viterbi(vec_Xi, mat_A, vec_lambda, vec_pi) :
    mat_emission  = get_mat_emission(vec_Xi, vec_lambda)
    num_of_states = len(mat_A)
    num_of_obs    = len(vec_Xi)
    mat_hs_seq    = np.zeros([num_of_states, num_of_obs])
    vec_logp_seq  = np.zeros(num_of_states)

    for j in range(0, num_of_states) :
        mat_hs_seq[j][0] = j
        if (vec_pi[j] * mat_emission[0][j] == 0) :
            vec_logp_seq[j] = -np.inf
        else :
            vec_logp_seq[j]  = math.log(vec_pi[j] * mat_emission[0][j]) / math.log(10)

    for n in range(1, num_of_obs) :
        mat_hs_seq_buf   = mat_hs_seq.copy()
        vec_logp_seq_buf = vec_logp_seq.copy()

        for j in range(0, num_of_states) :
            vec_h_logprob_i = np.zeros(num_of_states)
            for i in range(0, num_of_states) :
                vec_h_logprob_i[i] = vec_logp_seq[i] + math.log(mat_emission[n][j] * mat_A[i][j]) / math.log(10)

            max_element = max(vec_h_logprob_i)
            max_pos     = np.where(vec_h_logprob_i == max_element)[0][0]

            vec_logp_seq_buf[j] = max_element
            mat_hs_seq_buf[j]   = mat_hs_seq[max_pos].copy()

            mat_hs_seq_buf[j][n] = j

        mat_hs_seq   = mat_hs_seq_buf.copy()
        vec_logp_seq = vec_logp_seq_buf.copy()

    max_element = max(vec_logp_seq)

    max_pos = np.where(vec_logp_seq == max_element)[0][0]

    vec_hs_seq = mat_hs_seq[max_pos].copy()

    return vec_hs_seq
