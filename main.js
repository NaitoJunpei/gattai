// グラフサイズ
var x_base = 20;
var width_graph = 780;
var height_spike = 10;
var height_graph = 60;
var height_hist =54;
var max_repeat = 500;
var max_count = 10; //ヒストグラムを作る際に値をずらす回数
var res_graph = 200;

var spike_num;
var onset, offset;
var lv, np;

function LoadStart() {
  return new Promise(function(Main){
	document.getElementById("loading").style.visibility="visible";
	Main();
  });
}

function LoadEnd(){
	document.getElementById("loading").style.visibility="hidden";
}

// データの初期化
function ResetData() {
  //document.data.spikes.value = "0.049 0.141 0.225 0.274 0.303 0.320 0.336 0.437 0.478 0.496 0.538 0.553 0.562 0.580 0.632 0.633 0.645 0.659 0.663 0.673 0.678 0.700 0.721 0.728 0.750 0.765 0.771 0.792 0.815 0.838 0.853 0.867 0.905 0.923 0.936 0.947 0.990 1.003 1.021 1.052 1.073 1.106 1.112 1.141 1.153 1.170 1.185 1.213 1.215 1.279 1.285 1.338 1.417 1.462 1.587 1.591 1.764 1.871 1.888 1.944 1.965 2.006 2.013 2.021 2.046 2.063 2.105 2.126 2.136 2.164 2.216 2.288 2.291 2.308 2.393 2.424 2.436 2.471 2.559 2.580 2.602 2.643 2.674 2.718 2.762 2.909 2.947 2.979 3.010 3.032 3.039 3.052 3.103 3.157 3.185 3.217 3.242 3.273 3.326 3.337 3.348 3.375 3.392 3.415 3.426 3.449 3.475 3.489 3.545 3.636 3.656 3.669 3.696 3.718 3.749 3.780 3.849 3.862 3.869 3.935 4.084 4.202 4.222 4.249 4.287 4.348 4.365 4.386 4.416 4.428 4.431 4.442 4.459 4.492 4.501 4.520 4.534 4.550 4.595 4.603 4.634 4.639 4.655 4.662 4.678 4.691 4.710 4.725 4.732 4.753 4.766 4.805 4.820 4.868 4.887 4.891 4.931 4.965 4.991 5.012 5.072 5.083 5.111 5.181 5.220 5.277 5.327 5.431 5.494 5.565 5.838 5.863 5.894 6.014 6.086 6.103 6.119 6.137 6.149 6.168 6.186 6.214 6.264 6.278 6.306 6.353 6.414 6.422 6.450 6.517 6.532 6.598 6.666 6.693 6.711 6.743 6.788 6.803 6.838 6.846 6.863 6.876 6.891 6.909 6.952 6.959 6.976 6.996 7.015 7.028 7.039 7.052 7.057 7.092 7.130 7.148 7.165 7.195 7.226 7.230 7.241 7.247 7.275 7.287 7.302 7.311 7.317 7.326 7.340 7.354 7.381 7.407 7.440 7.466 7.517 7.519 7.583 7.645 7.658 7.676 7.689 7.778 7.788 7.832 7.864 7.884 7.973 8.042 8.167 8.523 8.592 8.644 8.724 8.776 8.809 8.842 8.863 8.892 8.965 8.969 8.981 9.001 9.012 9.025 9.055 9.060 9.085 9.099 9.136 9.168 9.206 9.212 9.237 9.256 9.294 9.301 9.309 9.330 9.367 9.397 9.448 9.565 9.630 9.683 9.736 9.783 9.845 9.867 9.914 9.950 9.985";
	document.data.spikes.value = "1.304 1.317 1.455 1.547 1.565 1.603 1.605 1.628 1.665 1.679 1.684 1.743 1.765 1.767 1.773 1.774 1.806 1.832 1.847 1.863 1.878 1.882 1.909 1.923 1.926 1.939 1.972 1.998 2.043 2.046 2.065 2.088 2.094 2.132 2.142 2.177 2.184 2.193 2.215 2.267 2.291 2.307 2.338 2.397 2.433 2.473 2.518 2.537 2.543 2.580 2.581 2.739 2.766 2.799 2.964 3.082 3.368 3.411 3.512 3.582 3.598 3.710 3.875 3.917 4.146 4.231 4.525 4.872 5.004 5.067 5.091 5.201 5.235 5.310 5.417 5.514 5.554 5.589 5.649 5.668 5.764 5.780 5.794 5.829 5.873 5.900 5.907 5.952 5.979 6.035 6.053 6.092 6.141 6.161 6.189 6.252 6.265 6.292 6.336 6.385 6.448 6.491 6.561 6.656 6.790 6.832 6.970 7.017 7.130 7.342 7.370 7.428 7.448 7.464 7.513 7.528 7.632 7.670 7.683 7.705 7.711 7.718 7.768 7.809 7.815 7.824 7.872 7.881 7.918 7.949 7.953 7.979 7.983 8.061 8.138 8.197 8.252 8.271 8.314 8.323 8.522 8.528 8.540 8.569 8.573 8.584 8.628 8.630 8.658 8.711 8.781 8.854 8.865 9.050 9.154 9.695 9.731 9.833 9.889 9.980";
	return 0;
}

// ランダムにデータを作成する関数
var MT = new MersenneTwister();
var Alpha = 2.0*Math.PI*MT.next();
var Beta  = 2.0*Math.PI*MT.next();
var Theta = 2.0*Math.PI*MT.next();
var Amp = 0.3+1.2*MT.next();

var SpikeData = new Array(3);

function RandomData() {
    var t1,t2;
    t1=Solve(0.0,Gamma(1.0));
    document.data.spikes.value = Number(t1.toFixed(3));
    var j=1;

    Alpha = 2.0*Math.PI*MT.next();
    Beta = 2.0*Math.PI*MT.next();
    Theta = 2.0*Math.PI*MT.next();
    Amp = 0.3+1.2*MT.next();
    
    var kappa = Math.random() * 1.25 + 0.75;
    while(1){
        t2=t1+Solve(t1,Gamma(kappa));
        if(t2>TIME) break;
        document.data.spikes.value += " " + t2.toFixed(3);
        t1=t2;
        j++;
    }
    return 0;
}
    
var Base=30.0;
var Amplitude=10.0;
var TIME=10.0;
var Period=[2.0/Math.PI,1.41421356/Math.PI,0.8989898/Math.PI];

function Rate_integral(prev_time,new_time){
    return Base*(new_time-prev_time) - Amplitude*Period[0]*Amp*( Math.cos(Alpha+new_time/Period[0]/Amp) - Math.cos(Alpha+prev_time/Period[0]/Amp) ) - Amplitude*Period[1]*Amp*( Math.cos(Beta+new_time/Period[1]/Amp) - Math.cos(Beta+prev_time/Period[1]/Amp) ) - Amplitude*Period[2]*Amp*( Math.cos(Theta+new_time/Period[2]/Amp) - Math.cos(Theta+prev_time/Period[2]/Amp) );
}

function Solve(prev_time,interval){
	var boundary = new Array(2);
	var new_interval;
	boundary[0]=0;	boundary[1]=0.5/Base;
	while( Rate_integral(prev_time,prev_time+boundary[1]) < interval ){		boundary[1]+=0.5/Base;		}
	
	while( boundary[1]-boundary[0] > Math.pow(10.0,-6.0) ){
		new_interval=0.5*(boundary[0]+boundary[1]);
		if( Rate_integral(prev_time,prev_time+new_interval) > interval )	boundary[1]=new_interval;
		else																boundary[0]=new_interval;	
	}
	new_interval=0.5*(boundary[0]+boundary[1]);
	if(new_interval<Math.pow(10.0,-8.0)) new_interval=Math.pow(10.0,-8.0);
	return new_interval;
}
function Gamma( kappa ){
    var int_kappa=Math.floor(kappa);
    var frac_kappa=kappa-Math.floor(kappa);
	var x_frac,x_int;
	/*integer part*/
	x_int=0;
	for(var i=0;i<int_kappa;i++){
		x_int+=-Math.log(MT.next());
	}
    /*fractional part*/
	if( frac_kappa < 0.01 ) x_frac=0;
	else{
		var b=(Math.exp(1.0)+frac_kappa)/Math.exp(1.0);
		while(1){
			var u=MT.next();
			var p=b*u;
			var uu=MT.next();
			if(p<=1.0){
				x_frac=Math.pow(p,1.0/frac_kappa);
				if(uu<=Math.exp(-x_frac)) break;
			}
			if(p>1.0){
				x_frac=-Math.log((b-p)/frac_kappa);
				if(uu<=Math.pow(x_frac,frac_kappa-1.0)) break;
			}
		}
	}
	return (x_int+x_frac)/kappa;
}

var time_old = new Array();
//現在のローカル時間が格納された、Date オブジェクトを作成する
var date_obj = new Date();

// メイン関数
function Main() {
  var spike_time = new Array();
  PostData(spike_time);

  spike_num = spike_time.length;
  // sort機能の実装を予定
  onset = spike_time[0] - 0.001 * (spike_time[spike_num - 1] - spike_time[0]);
  offset = spike_time[spike_num - 1] + 0.001 * (spike_time[spike_num - 1] - spike_time[0]);
  
  // 測定開始時に経過時間を変数に残す
  time_old[0] = new Date().getTime();
  SpikeRaster(spike_time);
  time_old[1] = new Date().getTime();
  DrawGraph_SSOS(spike_time);			// 旧法新法
  time_old[3] = new Date().getTime();
  DrawGraph_Kernel12(spike_time);	// カーネル法 & カーネル法(折り返し)
  time_old[5] = new Date().getTime();
  //DrawGraph_BayesNP(spike_time);	// ノンポアソンベイズ推定
  //time_old[5] = new Date().getTime();
  DrawGraph_Bayes(spike_time);	// ベイズ推定
  time_old[6] = new Date().getTime();
  DrawGraph_HMM(spike_time);		// 隠れマルコフモデル
  time_old[7] = new Date().getTime();
  
  document.getElementById("time").innerHTML = "<font size='2pt' face='Arial'>Spike Raster : " + (time_old[1]-time_old[0]) + " ms<br>(A) : " + (time_old[2]-time_old[1]) + " ms<br>(B)-(A) : " + (time_old[3]-time_old[2]) + " ms<br>(C) : " + (time_old[4]-time_old[3]) + " ms<br>(D)-(C) : " + (time_old[5]-time_old[4]) + " ms<br>(E) : " + (time_old[6]-time_old[5]) + " ms<br>(F) : " + (time_old[7]-time_old[6]) + " ms</font>";
  
  //DrawGraph(spike_time, SS(spike_time), "SS");  // 旧法
  //DrawGraph(spike_time, OS(spike_time), "OS");  // 新法
  //DrawGraph(spike_time, Kernel(spike_time), "Kernel");	// カーネル法
  //DrawGraph(spike_time, Kernel(spike_time), "Kernel2");	// カーネル法(折り返し)
  //DrawGraph(spike_time, 0, "HMM");	// 隠れマルコフモデル
}

// 入力データの処理 ソート機能はとりあえずカット
function PostData(spike_time) {
  var data_text = document.data.spikes.value.replace(/\r?\n/g," ").replace(/^\s+|\s+$/g,"");
  var data_seq = data_text.split(/[^0-9\.]+/);
  //document.data.spikes.value = ""
  for (var i = 0; i < data_seq.length; i++) {
    spike_time[i] = Number(data_seq[i]);
    // document.data.spikes.value += Math.round(spike_time[i]*1000) + " ";
  }
}

function SpikeRaster(spike_time){
	var names = ['SS','OS','Kernel','Kernel2','Bayes','HMM'];
	names.forEach(function(name){
		var wrap = d3.select('#raster_' + name);
		wrap.select("svg").remove();
		var svg = wrap.append("svg").attr("width",800).attr("height",15);
		
		var line = svg.append("line")
        .attr("x1",x_base)
        .attr("y1",height_spike)
        .attr("x2",x_base+width_graph)
        .attr("y2",height_spike)
        .attr("stroke","black")
        .attr("stroke-width",1);
		for (var i = 0; i < spike_num; i++) {
			    x = (spike_time[i] - onset) / (offset - onset);
			    var line = svg.append("line")
		        .attr("x1",x_base+width_graph * x)
		        .attr("y1",0)
		        .attr("x2",x_base+width_graph * x)
		        .attr("y2",height_spike)
		        .attr("stroke","black")
		        .attr("stroke-width",1);
		}
	});
}

// Shimazaki-Shinomoto, Omi-Shinomoto
function SSOS(spike_time) {
	var binsize;
	var count = new Array();
	var cost_SS, cost_OS, cost_SS_min, cost_OS_min;
	var w_av, av, va;
	var fano;
	var opt_binsize = new Array();	// [0]:SS, [1]:OS
	lv = 0;
	// lvの計算 パラメータも一緒に判断
	for (var i = 0; i < spike_num - 2; i++) {
		var interval = new Array(2);
	    interval[0] = spike_time[i + 1] - spike_time[i];
	    interval[1] = spike_time[i + 2] - spike_time[i + 1];
	    if ((interval[0] + interval[1]) != 0) lv += 3 * Math.pow(interval[0] - interval[1], 2.0) / Math.pow(interval[0] + interval[1], 2.0) / (spike_num - 2);
	    else lv += 3.0 / (spike_num - 2);
	}
	if (lv < 1) np = "regular";
	else np = "bursty";
	//	binの数を変化(最大500)　計算量的にはここを短縮したい気持ち

	var TT = spike_time.concat(spike_time.map(function(element) {
		return element + (offset - onset);
	}));
	for (var bin_num = 1; bin_num < max_repeat; bin_num++) {
		binsize = (offset - onset) / bin_num;
		for (var cost_count = 0; cost_count < max_count; cost_count++) {
			start = onset + cost_count * (offset - onset) / max_count;
			end = offset + cost_count * (offset - onset) / max_count;
			cost_SS = 0;
			cost_OS = 0;
			// スパイク数カウントの初期化
			for (i = 0; i < bin_num; i++) {
				count[i] = 0;
			}
			//スパイク数のカウント
			for (i = 0; TT[i] < end; i++) {
				if (TT[i] >= start) {
					count[Math.floor((TT[i] - start) / binsize)]++;
				}
			}
			// binのスパイク数の平均、分散を計算
			av = 0;
			va = 0;
			w_av = 0;
			for (i = 0; i < bin_num; i++) {
				if (count[i] > 2) {
					fano = 2.0 * lv / (3.0 - lv);
				} else {
					fano = 1.0;
				}
				w_av += fano * count[i] / bin_num;
				av += count[i] / bin_num;
				va += count[i] * count[i] / bin_num;
			}
			// コスト関数の計算
			cost_SS += (2.0 * av - (va - av * av)) / (binsize * binsize);
			cost_OS += (2.0 * w_av - (va - av * av)) / (binsize*binsize);
		}
		cost_SS /= max_count
		cost_OS /= max_count
		// コストが小さければ更新
		if (cost_SS < cost_SS_min || bin_num == 1) {
			cost_SS_min = cost_SS;
			opt_binsize[0] = binsize;
		}
		if (cost_OS < cost_OS_min || bin_num == 1) {
			cost_OS_min = cost_OS;
			opt_binsize[1] = binsize;
		}
	}
	return opt_binsize;
}

//カーネル法
function Kernel(spike_time){
	var width = new Array(50);
	var cost = new Array(width.length);
	var cost_min,min_index;
	for (var i=0; i<width.length; i++) {
		width[i] = (offset - onset) / (i+1);
		cost[i] = KernelCost(spike_time, width[i]);
		if(cost[i]<cost_min || i==0){
			cost_min = cost[i];
			min_index = i;
		}
	}
	return width[min_index];
}

// カーネル法コスト関数
function KernelCost(spike_time, width) {
	var A = 0; 
	for (var i=0; i<spike_time.length; i++) {
		for (var j=i+1; j<spike_time.length; j++) {
			var x = spike_time[i]-spike_time[j];
			if (x < 5*width) {
				A = A + 2*Math.exp(-x*x/4/width/width) - 4*Math.sqrt(2)*Math.exp(-x*x/2/width/width);
			}
		}
	}
	return (spike_time.length/width + A/width) / 2 / Math.sqrt(Math.PI);
}

function Bayes(spike_time){
	var n=0;	// 10^n < x < 10^(n+1)
	if(offset-onset>1){
		while((spike_time[spike_time.length-1]-spike_time[0])>Math.pow(10,n+1)){
			n += 1;
		}
	}else{
		while((spike_time[spike_time.length-1]-spike_time[0])<Math.pow(10,n)){
			n -= 1;
		}
	}
}

function DrawGraph_SSOS(spike_time){
	//SS
	var wrap = d3.select('#graph_SS');
	wrap.select("svg").remove();	// 初期化
	var svg = wrap.append("svg").attr("width",x_base+width_graph).attr("height",height_graph);
	
	var opt = new Array();
	opt = SSOS(spike_time);
	var opt_rate_SS = new Array();
	var rate_max = EstimateRate(spike_time, opt[0], opt_rate_SS);
	var x,y,xx,yy;
	for (var i = 0; i < opt_rate_SS.length; i++) {
	    x = i * opt[0] / (offset - onset);
	    y = opt_rate_SS[i] / rate_max;
	    xx = x_base + width_graph * x;
	    yy = height_hist * y;
	    if (onset + (i + 1) * opt[0] < offset){
	    	svg.append("rect").attr("x", xx).attr("y", height_graph-yy).attr("width", width_graph * opt[0] / (offset - onset)).attr("height", yy).attr("fill","#87CEFA").attr("stroke","#67AEDA");
	    }else{
	    	svg.append("rect").attr("x", xx).attr("y", height_graph-height_hist * y).attr("width", width_graph - width_graph * x).attr("height", height_hist * y).attr("fill","#87CEFA").attr("stroke","#67AEDA");
	    }

	}
	svg.append("rect").attr("x", x_base).attr("y", 0).attr("width", width_graph).attr("height", height_graph).attr("stroke","black").attr("stroke-width",1).attr("fill","none");
	document.getElementById("optimal_SS").innerHTML = "Optimal bin size = <font color=\"red\">" + opt[0].toFixed(2) + "</font>";

	time_old[2] = new Date().getTime();
	
	//OS
	wrap = d3.select('#graph_OS');
	wrap.select("svg").remove();	// 初期化
	svg = wrap.append("svg").attr("width",x_base+width_graph).attr("height",height_graph);
	
	var opt_rate_OS = new Array();
	rate_max = EstimateRate(spike_time, opt[1], opt_rate_OS);
	for (var i = 0; i < opt_rate_OS.length; i++) {
	    x = i * opt[1] / (offset - onset);
	    y = opt_rate_OS[i] / rate_max;
	    xx = x_base + width_graph * x;
	    yy = height_hist * y;
	    if (onset + (i + 1) * opt[1] < offset){
	    	svg.append("rect").attr("x", xx).attr("y", height_graph-yy).attr("width", width_graph * opt[1] / (offset - onset)).attr("height", yy).attr("fill","#7FFFD4").attr("stroke","#5FDFB4");
	    }else{
	    	svg.append("rect").attr("x", xx).attr("y", height_graph-height_hist * y).attr("width", width_graph - width_graph * x).attr("height", height_hist * y).attr("fill","#7FFFD4").attr("stroke","#5FDFB4");
	    }
	}
	svg.append("rect").attr("x", x_base).attr("y", 0).attr("width", width_graph).attr("height", height_graph).attr("stroke","black").attr("stroke-width",1).attr("fill","none");

	document.getElementById("optimal_OS").innerHTML = "Optimal bin size = <font color=\"red\">" + opt[1].toFixed(2) + "</font>&nbsp;&nbsp;&nbsp;&nbsp;Irregularity is estimated as Lv = <font color=\"red\">" + lv.toFixed(2) + "</font>";
}

function DrawGraph_Kernel12(spike_time){
	//	Kernel(C)
	var wrap = d3.select('#graph_Kernel');
	wrap.select("svg").remove();	// 初期化
	var svg = wrap.append("svg").attr("width",x_base+width_graph).attr("height",height_graph);
	
	var opt = Kernel(spike_time);
	var opty1 = new Array();
	var opty2 = new Array();
	var maxy = kern12(spike_time, opt, opty1, opty2);

	var xy1 = new Array();
	for (var i = 0;i<res_graph;i++) {
		xy1[i] = [x_base + Math.round(i*width_graph/(res_graph-1)), height_graph - Math.round(height_graph*opty1[i]/(1.2*maxy))];
	}
	xy1.unshift([x_base, height_graph]);
	xy1.push([x_base+width_graph, height_graph]);
	var line = d3.svg.line()
	      .x(function(d) {return d[0];})
	      .y(function(d) {return d[1];});
	svg.append("path").attr("d", line(xy1) ).attr("fill","#F0E68C").attr("stroke","#D0C66C");
	svg.append("rect").attr("x", x_base).attr("y", 0).attr("width", width_graph).attr("height", height_graph).attr("stroke","black").attr("stroke-width",1).attr("fill","none");
	document.getElementById("optimal_Kernel").innerHTML = "Optimal bandwidth = <font color=\"red\">" + opt.toFixed(2) + "</font>";
	
	time_old[4] = new Date().getTime();
	
	// Kernel2(D)
	var wrap = d3.select('#graph_Kernel2');
	wrap.select("svg").remove();	// 初期化
	var svg = wrap.append("svg").attr("width",x_base+width_graph).attr("height",height_graph);

	var xy2 = new Array();
	for (var i = 0;i<res_graph;i++) {
		xy2[i] = [x_base + Math.round(i*width_graph/(res_graph-1)), height_graph - Math.round(height_graph*opty2[i]/(1.2*maxy))];
	}
	xy2.unshift([x_base, height_graph]);
	xy2.push([x_base+width_graph, height_graph]);
	var line = d3.svg.line()
	      .x(function(d) {return d[0];})
	      .y(function(d) {return d[1];});
	svg.append("path").attr("d", line(xy2) ).attr("fill","#FFDEAD").attr("stroke","#DFBE8D");
	svg.append("rect").attr("x", x_base).attr("y", 0).attr("width", width_graph).attr("height", height_graph).attr("stroke","black").attr("stroke-width",1).attr("fill","none");
	document.getElementById("optimal_Kernel2").innerHTML = "Optimal bandwidth = <font color=\"red\">" + opt.toFixed(2) + "</font>";
}

function DrawGraph_HMM(spike_time){
	var wrap = d3.select('#graph_HMM');
	wrap.select("svg").remove();	// 初期化
	var svg = wrap.append("svg").attr("width",x_base+width_graph).attr("height",height_graph);
	
	var x,y,maxy;
	var opty;
	var opt = ((offset-onset)/(spike_time.length-1)) * 5;	// step width = ISI * 5
	opty = get_hmm_ratefunc(spike_time, opt);
	for(var i=0; i<opty.length; i++){
		if(i==0 || maxy<opty[i][1]) maxy=opty[i][1];
	}
	var x,y,xx,yy;
	for (var i = 0; i < opty.length; i++) {
		var x_pos=x_base+i*width_graph/opty.length;
		var height=height_hist*opty[i][1]/maxy;
	    if (onset + i * opt < offset){
	    	svg.append("rect").attr("x", x_pos).attr("y", height_graph-height).attr("width", width_graph/opty.length+1).attr("height", height).attr("fill","#DA75F3");
	    }else{
			x_pos=offset;
	    	svg.append("rect").attr("x", x_pos).attr("y", height_graph-height).attr("width", width_graph/opty.length+1).attr("height", height).attr("fill","#DA75F3");
	    }
	}
	svg.append("rect").attr("x", x_base).attr("y", 0).attr("width", width_graph).attr("height", height_graph).attr("stroke","black").attr("stroke-width",1).attr("fill","none");
}

function DrawGraph_Bayes(spike_time){
	var wrap = d3.select('#graph_Bayes');
	wrap.select("svg").remove();	// 初期化
	var svg = wrap.append("svg").attr("width",x_base+width_graph).attr("height",height_graph);

	var maxy;
	var xy = new Array();
	
	var kalman_data = SecondStage(spike_time);
	// ThirdStage(spike_time,beta);
	for(var i=0; i<kalman_data[0].length; i++){
		if(i==0 || maxy<kalman_data[0][i]) maxy=kalman_data[0][i];
	}
	for (var i = 0;i<spike_time.length-1;i++) {
		xy[i] = [x_base + width_graph*(spike_time[i]/2+spike_time[i+1]/2-spike_time[0])/(spike_time[spike_time.length-1]-spike_time[0]), height_graph - height_graph*kalman_data[0][i]/(1.2*maxy)];
	}
	xy.unshift([x_base, height_graph - height_graph*kalman_data[0][0]/(1.2*maxy)]);
	xy.unshift([x_base, height_graph]);
	xy.push([x_base+width_graph, height_graph - height_graph*kalman_data[0][spike_time.length-2]/(1.2*maxy)]);
	xy.push([x_base+width_graph, height_graph]);
	var line = d3.svg.line()
	      .x(function(d) {return d[0];})
	      .y(function(d) {return d[1];});
	svg.append("path").attr("d", line(xy) ).attr("fill","#FFC0CB").attr("stroke","#DFA0AB");
	svg.append("rect").attr("x", x_base).attr("y", 0).attr("width", width_graph).attr("height", height_graph).attr("stroke","black").attr("stroke-width",1).attr("fill","none");
}


function kern12(spike_time, width, y1, y2) {
	var x = new Array(res_graph)
	x[0] = onset;
	
	var maxy=0;
	var gauss;
	var addNumber = 0;
	
	for (var i=0; i<res_graph; i++) {
		x[i+1] = x[i] + (offset-onset)/(res_graph-1); 
	}
	var maxy=0;
	var gauss;
	for (var i=0; i<res_graph; i++) {
		addNumber = 0;
		y1[i] = 0;
		for (var j in spike_time) {
			if((x[i]-5*width <= spike_time[j]) && (spike_time[j] <= x[i]+5*width)){
				gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-spike_time[j])*(x[i]-spike_time[j])/2/width/width);
				y1[i] = y1[i] + gauss / spike_time.length;
			}
			if (x[i]-5*width < onset) {
				if (-(x[i]-5*width)+2*onset > spike_time[j]){
					gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-(onset-(spike_time[j]-onset)))*(x[i]-(onset-(spike_time[j]-onset)))/2/width/width);
					addNumber = addNumber + gauss / spike_time.length;
				}
			}else if(x[i]+5*width > offset){
				if(-(x[i]+5*width)+2*offset > spike_time[i]){
					gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-(offset+(offset-spike_time[j])))*(x[i]-(offset+(offset-spike_time[j])))/2/width/width);
					addNumber = addNumber + gauss / spike_time.length;
				}
			}
		}
		y2[i] = y1[i] + addNumber;
		if(maxy<y2[i]) maxy=y2[i];
	}
	return maxy;
}

// binsizeとbinrateからレートを推定
function EstimateRate(spike_time, opt_binsize, opt_rate) {
  var opt_binnum = Math.ceil((spike_time[spike_num - 1] - onset) / opt_binsize);
  var rate_max;

  for (var i = 0; i < opt_binnum; i++) {
    opt_rate[i] = 0;
  }
  for (i = 0; i < spike_num; i++) {
    opt_rate[Math.floor((spike_time[i] - onset) / opt_binsize)] += 1.0 / opt_binsize;
  }
  for (i = 0; i < opt_binnum; i++) {
    if (i == 0 || opt_rate[i] > rate_max) rate_max = opt_rate[i];
  }
  return rate_max;
}
// データ出力
function GenerateOutputFileMessage(message) {
	return "<div id='Output'></div> <script type='text/javascript'>var myBlob = new Blob([\"" + message + "\"], {type: 'text/html'}); var url = URL.createObjectURL(myBlob); document.getElementById('Output').innerHTML = '<a href=' + url + ' download=datasheet.csv>download as csv</a>';</script>";
}

function OutputResults_SS() {
	var result;
	var spike_time = new Array();
	PostData(spike_time);
	var opt_binsize = new Array();
	var opt_rate = new Array();
	opt_binsize = SSOS(spike_time);
	EstimateRate(spike_time, opt_binsize[0], opt_rate);

	//save as csv
	var filemessage = "X-AXIS,Y-AXIS\\n";
	filemessage += onset.toFixed(2) + ",0\\n";
	for (var i = 0; i < opt_rate.length; i++) {
		filemessage += (onset + i * opt_binsize[0]).toFixed(2) + "," + opt_rate[i].toFixed(2) + "\\n";
		filemessage += (onset + (i + 1) * opt_binsize[0]).toFixed(2) + "," + opt_rate[i].toFixed(2) + "\\n";
	}
	filemessage += (onset + opt_rate.length * opt_binsize[0]).toFixed(2) + ",0\\n";

	WIN_RESULTS = window.open();
	WIN_RESULTS.document.open();
	WIN_RESULTS.document.writeln("<title>Data Sheet of the Optimized Histogram</title>");
	WIN_RESULTS.document.writeln("<h2>Histgram: Poissonian optimization</h2>");
	WIN_RESULTS.document.writeln("Optimal binsize: <b>"+opt_binsize[0].toFixed(2)+"</b><br><br>");

	WIN_RESULTS.document.writeln(GenerateOutputFileMessage(filemessage));

	WIN_RESULTS.document.writeln("<table border=1><tr align=center><td width=150> X-AXIS (time)  </td><td width=150> Y-AXIS (density)</td>");
	WIN_RESULTS.document.writeln("<tr align=right><td>" + onset.toFixed(2) + "</td><td>0.00</td></tr>");
	for (var i=0;i<opt_rate.length;i++) {
		WIN_RESULTS.document.writeln("<tr align=right><td>" + (onset + i * opt_binsize[0]).toFixed(2) + "</td><td>" + opt_rate[i].toFixed(2) + "</td></tr>");
		WIN_RESULTS.document.writeln("<tr align=right><td>" + (onset + (i + 1) * opt_binsize[0]).toFixed(2) + "</td><td>" + opt_rate[i].toFixed(2) + "</td></tr>");
	}
	WIN_RESULTS.document.writeln("<tr align=right><td>" + (onset + opt_rate.length * opt_binsize[0]).toFixed(2) + "</td><td>0.00</td></tr>");
	WIN_RESULTS.document.writeln("</table><br>");
	WIN_RESULTS.document.close();
}

function OutputResults_OS() {
	var result;
	var spike_time = new Array();
	PostData(spike_time);
	var opt_binsize = new Array();
	var opt_rate = new Array();
	opt_binsize = SSOS(spike_time);
	EstimateRate(spike_time, opt_binsize[1], opt_rate);

	//save as csv
	var filemessage = "X-AXIS,Y-AXIS\\n";
	filemessage += onset.toFixed(2) + ",0\\n";
	for (var i = 0; i < opt_rate.length; i++) {
		filemessage += (onset + i * opt_binsize[1]).toFixed(2) + "," + opt_rate[i].toFixed(2) + "\\n";
		filemessage += (onset + (i + 1) * opt_binsize[1]).toFixed(2) + "," + opt_rate[i].toFixed(2) + "\\n";
	}
	filemessage += (onset + opt_rate.length * opt_binsize[1]).toFixed(2) + ",0\\n";

	
	WIN_RESULTS = window.open();
	WIN_RESULTS.document.open();
	WIN_RESULTS.document.writeln("<title>Data Sheet of the Optimized Histogram</title>");
	WIN_RESULTS.document.writeln("<h2>Histgram: Non-Poissonian optimization</h2>");
	WIN_RESULTS.document.writeln("Optimal binsize: <b>"+opt_binsize[1].toFixed(2)+"</b><br>Lv: <b>" + lv.toFixed(2) + "</b> (" + np + " firing)<br><br>");

	WIN_RESULTS.document.writeln(GenerateOutputFileMessage(filemessage)); 

	WIN_RESULTS.document.writeln("<table border=1><tr align=center><td width=150> X-AXIS (time)  </td><td width=150> Y-AXIS (density) </td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>" + onset.toFixed(2) + "</td><td>0.00</td></tr>");
	for (var i=0;i<opt_rate.length;i++) {
		WIN_RESULTS.document.writeln("<tr align=right><td>" + (onset + i * opt_binsize[1]).toFixed(2) + "</td><td>" + opt_rate[i].toFixed(2) + "</td></tr>");
		WIN_RESULTS.document.writeln("<tr align=right><td>" + (onset + (i + 1) * opt_binsize[1]).toFixed(2) + "</td><td>" + opt_rate[i].toFixed(2) + "</td></tr>");
	}
	WIN_RESULTS.document.writeln("<tr align=right><td>" + (onset + opt_rate.length * opt_binsize[1]).toFixed(2) + "</td><td>0.00</td></tr>");
	WIN_RESULTS.document.writeln("</table><br>");
	WIN_RESULTS.document.close();
}

function xaxisForKernel(spike_time) {
	var x = new Array(res_graph);
	var data_max = spike_time[spike_time.length - 1];
	var data_min = spike_time[0];
	x[0] = data_min;
	for (var i = 0; i < res_graph - 1; i++) {
		x[i + 1] = x[i] + (data_max - data_min) / (res_graph - 1);
	}
	return x;
}

function kern(spike_time, width, y) {
	var x = new Array(res_graph)
	x[0] = onset;
	for (var i=0; i<res_graph; i++) {
		x[i+1] = x[i] + (offset-onset)/(res_graph-1); 
	}
	var maxy=0;
	var gauss;
	for (var i=0; i<res_graph; i++) {
		y[i] = 0;
		for (var j in spike_time) {
			if((x[i]-5*width <= spike_time[j]) && (spike_time[j] <= x[i]+5*width)){
				gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-spike_time[j])*(x[i]-spike_time[j])/2/width/width);
				y[i] = y[i] + gauss;
			}
		}
		if(maxy<y[i]) maxy=y[i];
	}
	return maxy;
}

function kern2(spike_time, width, y) {
	var x = new Array(res_graph)
	x[0] = onset;
	for (var i=0; i<res_graph; i++) {
		x[i+1] = x[i] + (offset-onset)/(res_graph-1); 
	}
	var maxy=0;
	var gauss;
	var addNumber = 0;

	for (var i=0; i<res_graph; i++) {
		addNumber = 0;
		y[i] = 0;
		for (var j in spike_time) {
			if((x[i]-5*width <= spike_time[j]) && (spike_time[j] <= x[i]+5*width)){
				gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-spike_time[j])*(x[i]-spike_time[j])/2/width/width);
				y[i] = y[i] + gauss;
			}
		
				
			if (x[i] - 5*width<onset) {
				if (-(x[i]-5*width)+2*onset > spike_time[j]){
					gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-(onset-(spike_time[j]-onset)))*(x[i]-(onset-(spike_time[j]-onset)))/2/width/width);
					addNumber = addNumber + gauss;
				}
			}else if(x[i]+5*width>offset){
				if(-(x[i]+5*width)+2*offset > spike_time[i]){
					gauss = 1/Math.sqrt(2*Math.PI)/width*Math.exp(-(x[i]-(offset+(offset-spike_time[j])))*(x[i]-(offset+(offset-spike_time[j])))/2/width/width);
					addNumber = addNumber + gauss;
				}
			}
		}
		y[i] += addNumber;
		if(maxy<y[i]) maxy=y[i];
	}
	console.log(y);
	return maxy;
}


function OutputResults_Kernel() {
	var spike_time = new Array();
	PostData(spike_time);
	var opt = Kernel(spike_time);
	var opty = new Array();
	kern(spike_time, opt, opty);
	var xaxis = xaxisForKernel(spike_time);

	//save as csv
	var filemessage = "X-AXIS,Y-AXIS\\n";
	filemessage += xaxis[0].toFixed(3) + ",0\\n";
	for (var i = 0; i < xaxis.length; i++) {
		filemessage += xaxis[i].toFixed(3) + "," + opty[i].toFixed(3) + "\\n";
	}
	filemessage += xaxis[xaxis.length - 1].toFixed(3) + ",0\\n";

	
	WIN_RESULTS = window.open();
	WIN_RESULTS.document.open();
	WIN_RESULTS.document.writeln("<title>Data Sheet of the Optimized Histogram</title>");
	WIN_RESULTS.document.writeln("<h2>Histgram: Kernel Density Estimation</h2>");
	WIN_RESULTS.document.writeln("Optimal Bandwidth: <b>"+opt.toFixed(3)+"</b><br><br>");

	WIN_RESULTS.document.writeln(GenerateOutputFileMessage(filemessage));
	
	WIN_RESULTS.document.writeln("<table border=1><tr align=center><td width=150> X-AXIS (time)  </td><td width=150> Y-AXIS (density) </td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>"+xaxis[0].toFixed(3)+"</td><td>0.00</td></tr>");
	for (var i=0;i<xaxis.length;i++) {
		WIN_RESULTS.document.writeln("<tr align=right><td>"+xaxis[i].toFixed(3)+"</td><td>" + opty[i].toFixed(3) + "</td></tr>");
	}
	WIN_RESULTS.document.writeln("<tr align=right><td>"+xaxis[xaxis.length-1].toFixed(3)+"</td><td>0.00</td></tr>");
	WIN_RESULTS.document.writeln("</table><br>");
	WIN_RESULTS.document.close();
}

function OutputResults_Kernel2() {
	var spike_time = new Array();
	PostData(spike_time);
	var opt = Kernel(spike_time);
	var opty = new Array();
	kern2(spike_time, opt, opty);
	var xaxis = xaxisForKernel(spike_time);

	//save as csv
	var filemessage = "X-AXIS,Y-AXIS\\n";
	filemessage += xaxis[0].toFixed(3) + ",0\\n";
	for (var i = 0; i < xaxis.length; i++) {
		filemessage += xaxis[i].toFixed(3) + "," + opty[i].toFixed(3) + "\\n";
	}
	filemessage += spike_time[spike_time.length - 1] + ",0\\n";
	
	WIN_RESULTS = window.open();
	WIN_RESULTS.document.open();
	WIN_RESULTS.document.writeln("<title>Data Sheet of the Optimized Histogram</title>");
	WIN_RESULTS.document.writeln("<h2>Histgram: Kernel Density Estimation</h2>");
	WIN_RESULTS.document.writeln("Optimal Bandwidth: <b>"+opt.toFixed(3)+"</b><br><br>");
	
	WIN_RESULTS.document.writeln(GenerateOutputFileMessage(filemessage));
	
	WIN_RESULTS.document.writeln("<table border=1><tr align=center><td width=150> X-AXIS (time)  </td><td> Y-AXIS (density) </td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>"+xaxis[0].toFixed(3)+"</td><td>0.00</td></tr>");
	for (var i=0;i<xaxis.length;i++) {
		WIN_RESULTS.document.writeln("<tr align=right><td>"+xaxis[i].toFixed(3)+"</td><td>" + opty[i].toFixed(3) + "</td></tr>");
	}
	WIN_RESULTS.document.writeln("<tr align=right><td>"+xaxis[xaxis.length -1].toFixed(3)+"</td><td>0.00</td></tr>");
	WIN_RESULTS.document.writeln("</table><br>");
	WIN_RESULTS.document.close();
}

function OutputResults_HMM() {
	var spike_time = new Array();
	PostData(spike_time);
	var opty;
	var opt = (offset-onset)/(spike_time.length-1);
	var time = onset;
	opty = get_hmm_ratefunc(spike_time, opt);	//描画の細かさ0.05 ?

	//save as csv
	var filemessage = "X-AXIS,Y-AXIS\\n";
	filemessage += time.toFixed(3) + ",0\\n";
	filemessage += time.toFixed(3) + "," + opty[0][1].toFixed(3) + "\\n";
	time += opt;
	for (var i = 1; i < opty.length; i++) {
		if (opty[i][1] != opty[i - 1][1]) {
			filemessage += time.toFixed(3) + "," + opty[i - 1][1].toFixed(3) + "\\n";
			filemessage += time.toFixed(3) + "," + opty[i][1].toFixed(3) + "\\n";
		}
		time += opt;
	}
	filemessage += time.toFixed(3) + "," + opty[opty.length - 1][1].toFixed(3) + "\\n";
	filemessage += time.toFixed(3) + ",0\\n";
	time = onset;
		   
	WIN_RESULTS = window.open();
	WIN_RESULTS.document.open();
	WIN_RESULTS.document.writeln("<title>Data Sheet of the Optimized Histogram</title>");
	WIN_RESULTS.document.writeln("<h2>Histgram: Two state hidden Markov model</h2>");

	WIN_RESULTS.document.writeln(GenerateOutputFileMessage(filemessage));

	WIN_RESULTS.document.writeln("<table border=1><tr align=center><td width=150> X-AXIS (time)  </td><td width=150> Y-AXIS (density) </td></tr>");
	//WIN_RESULTS.document.writeln("<tr align=right><td>0.000</td><td>0.000</td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>" +time.toFixed(2)+"</td><td>0.000</td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>" + time.toFixed(2) + "</td><td>" + opty[0][1].toFixed(3) + "</td></tr>");
	time+=opt;
	for (var i=1;i<opty.length;i++) {
		if(opty[i][1]!=opty[i-1][1]){
			WIN_RESULTS.document.writeln("<tr align=right><td>"+time.toFixed(2)+"</td><td>" + opty[i-1][1].toFixed(3)+"</td></tr>");
			WIN_RESULTS.document.writeln("<tr align=right><td>"+time.toFixed(2)+"</td><td>" + opty[i][1].toFixed(3) + "</td></tr>");
		}
		time+=opt;
	}
	WIN_RESULTS.document.writeln("<tr align=right><td>"+ time.toFixed(2) +"</td><td>" + opty[opty.length-1][1].toFixed(3) + "</td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>"+ time.toFixed(2) +"</td><td>0.000</td></tr>");
	WIN_RESULTS.document.writeln("</table><br>");
	WIN_RESULTS.document.writeln("</blockquote>");
	WIN_RESULTS.document.close();
}

function OutputResults_Bayes(){
	var spike_time = new Array();
	PostData(spike_time);
	var opty;
	var kalman_data = SecondStage(spike_time);
	// ThirdStage(spike_time,beta);

	//save as csv
	var filemessage = "X-AXIS,Y-AXIS\\n";
	filemessage += ((spike_time[0] + spike_time[1]) / 2).toFixed(3) + ",0\\n";
	for (var i = 0; i < spike_time.length - 1; i++) {
		filemessage += ((spike_time[i] + spike_time[i + 1]) / 2).toFixed(3) + "," + kalman_data[0][i].toFixed(3) + "\\n";
	}
	filemessage += ((spike_time[spike_time.length - 2] + spike_time[spike_time.length - 1]) / 2).toFixed(3) + ",0\\n";

	WIN_RESULTS = window.open();
	WIN_RESULTS.document.open();
	WIN_RESULTS.document.writeln("<title>Data Sheet of the Optimized Histogram</title>");
	WIN_RESULTS.document.writeln("<h2>Histgram: Bayesian model Estimation</h2>");
	WIN_RESULTS.document.writeln("<br><br>");

	WIN_RESULTS.document.writeln(GenerateOutputFileMessage(filemessage));
	
	WIN_RESULTS.document.writeln("<table border=1><tr align=center><td width=150> X-AXIS (time)  </td><td width=150> Y-AXIS (density) </td></tr>");
	WIN_RESULTS.document.writeln("<tr align=right><td>"+((spike_time[0] + spike_time[1]) / 2).toFixed(3)+"</td><td>0.00</td></tr>");
	for (var i=0;i<spike_time.length - 1;i++) {
		WIN_RESULTS.document.writeln("<tr align=right><td>"+((spike_time[i] + spike_time[i + 1]) / 2).toFixed(3)+"</td><td>" + kalman_data[0][i].toFixed(3) + "</td></tr>");
	}
	WIN_RESULTS.document.writeln("<tr align=right><td>"+((spike_time[spike_time.length - 2] + spike_time[spike_time.length - 1]) / 2).toFixed(3) + "</td><td>0.00</td></tr>");
	WIN_RESULTS.document.writeln("</table><br>");
	WIN_RESULTS.document.close();
}
