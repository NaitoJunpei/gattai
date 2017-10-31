// カルマンフィルタを用いた計算
var EL = new Array();
EL[0] = new Array();
EL[1] = new Array();
var VL = new Array();
VL[0] = new Array();
VL[1] = new Array();
var EL_N = new Array();
var VL_N = new Array();
var COVL_N = new Array();
var N;

/*
SecondStage関数
EMアルゴリズムによってパラメータbetaを最適化し、そのパラメータbetaを使ってカルマンフィルタでスパイクの発生率を推定する.

引数
spike_time: スパイク列

返り値
kalman_data: スパイクの発生率

内部変数
mu: 平均発火率
beta0: ハイパーパラメータbetaの初期値
beta: EMアルゴリズムによって最適化されたハイパーパラメータbeta
kalman_data: 推定されたスパイクの発生率
*/

function SecondStage(spike_time){
	var mu = spike_time.length / (spike_time[spike_time.length - 1] - spike_time[0]);	// 平均発火率
	var beta0 = Math.pow(mu,-3);
	var beta = EMmethod(spike_time,beta0);
	var kalman_data = KalmanFilter(spike_time,beta);
	
	return kalman_data;
}
/*
function ThirdStage(spike_time,beta){
	NGEMInitialize(spike_time,beta);
	beta = NGEMmethod();
	var nongaussian_data = NGF(beta);
	var D = NGFD();// 時刻数
	var dt = NGFdt();
	return nongaussian_data;
}
*/

/*
EMmethod関数
EMアルゴリズムを利用してパラメータbetaの推定を行う. パラメータ推定の内部でカルマンフィルタを利用している.

引数
spike_time: スパイク列
beta0: パラメータの初期値

返り値
beta2: 推定されたパラメータbeta

内部変数
beta1: 更新前のパラメータの値
beta2: 更新後のパラメータの値
T0: (データの不具合により)同じスパイクが2回以上記述されていた場合に結果を補正するための値
kalman: スパイク列にカルマンフィルタを適用した値
*/

function EMmethod(spike_time,beta0){
	KFinitialize(spike_time);
	var beta1 = 0;
	var beta2 = beta0;

	var T0;
	for(var j=0;j<100;j++){
		beta1 = beta2;
		var kalman = KalmanFilter(spike_time, beta1);
		beta2 = 0;
		T0=0;
		for(var i=0; i<N-1; i++){
			if(spike_time[i+1]-spike_time[i]>0){
				beta2 += (kalman[1][i+1]+kalman[1][i]-2*kalman[2][i]+(kalman[0][i+1]-kalman[0][i])*(kalman[0][i+1]-kalman[0][i]))/(spike_time[i+1]-spike_time[i]);
			}else{
				T0 += 1;	// interspike interval がゼロのものがあったときの補正
			}
		}
	        beta2 = (N-T0-1)/(2*beta2);
	}
	return beta2;
}

/*
KFinitialize関数
カルマンフィルタで使用するNの値, EL, VLの初期値を設定する.

引数
spike_time: スパイク列

返り値
なし

内部変数
mu: 平均発火率の逆数
IEL: muと同じ
IVL: muの1/3の二乗
*/
function KFinitialize(spike_time){
	N = spike_time.length - 1;
	// N = interspike interval length
	var mu = 0;
	for(var i=0;i<N;i++){
		mu += spike_time[i+1]-spike_time[i];
	}
	mu = N/mu;
	// filtering
	var IEL = mu;
	var IVL = (mu/3)*(mu/3);
	var A = IEL - (spike_time[1]-spike_time[0])*IVL;
	EL[0][0] = (A+Math.sqrt(A*A+4*IVL))/2;
	VL[0][0] = 1/(1/IVL+1/(EL[0][0]*EL[0][0]));
}

/*
KalmanFilter関数
カルマンフィルタを利用してスパイクの発生率を推定する.

引数
spike_time: スパイク列
beta: ハイパーパラメータbeta

返り値
outdata: 推定したスパイクの発生率.
outdata[0]が値, [1]がその分散, [2]が共分散となっている.
*/

function KalmanFilter(spike_time,beta){
	for(var i=0;i<N-1;i++){
		EL[1][i]=EL[0][i];
		VL[1][i]=VL[0][i]+(spike_time[i+1]-spike_time[i])/(2*beta);
		A=EL[1][i]-(spike_time[i+2]-spike_time[i+1])*VL[1][i];
		EL[0][i+1]=(A+Math.sqrt(A*A+4*VL[1][i]))/2;
		VL[0][i+1]=1/(1/VL[1][i]+1/(EL[0][i+1]*EL[0][i+1]));
	}
	EL_N[N-1] = EL[0][N-1];
	VL_N[N-1] = VL[0][N-1];
	var H = new Array();
	for(var i=N-2;i>=0;i--){
		H[i] = VL[0][i]/VL[1][i];
		EL_N[i]=EL[0][i]+H[i]*(EL_N[i+1]-EL[1][i]);
		VL_N[i]=VL[0][i]+H[i]*H[i]*(VL_N[i+1]-VL[1][i]);
		COVL_N[i]=H[i]*VL_N[i+1];
	}
	var outdata = new Array();
	outdata[0] = new Array();
	outdata[1] = new Array();
	outdata[2] = new Array();
	for(var i=0;i<N;i++){
		outdata[0][i]=EL_N[i];
		outdata[1][i]=VL_N[i];
		outdata[2][i]=COVL_N[i];
	}
	return outdata;
}
/*
function NGEMmethod(spike_time, beta){
	
}

*/
