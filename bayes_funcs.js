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
