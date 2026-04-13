clear; clc; close all;

%% ECE466_SS26__Honors_Chenxin_181333637 
%% Dr.Panos
load('djiaw_2019.mat');          
dates = djiaw_total(:,1);        
djia  = djiaw_total(:,2);        
Ntotal = length(djia);          
%  T.1.a Plot + Investment
r = 0.03;                        %APR 3%
bank_weekly = 1 + r/52;          
initial_money = 1000;            %Starting Money

figure;
plot(1:Ntotal,djia,'LineWidth',1.2);
xlabel('Week');
ylabel('DJIA');
title('DJIA (Linear Scale)');
grid on;

figure;
semilogy(1:Ntotal,djia,'LineWidth',1.2);
xlabel('Week');
ylabel('DJIA');
title('DJIA (Semi-log Scale)');
grid on;

money_djia_total = initial_money * djia(end)/djia(1);
money_bank_total = initial_money * bank_weekly^Ntotal;
apr_equiv_total = 52 * ((money_djia_total/initial_money)^(1/Ntotal) - 1);
fprintf('T.1.a\n');
fprintf('Total weeks = %d\n', Ntotal);
fprintf('DJIA final money = %.6f\n', money_djia_total);
fprintf('Bank final money = %.6f\n', money_bank_total);
fprintf('Equivalent APR = %.6f\n\n', apr_equiv_total);

%  T.1.b Build X and solve a

p = 3;                
N = 520;              

xdata = djia(1:N);   
x = xdata(p+1:N);     

%% Token from Chat GPT how to generate the matrix x and vector in Matlab and how to use loop to use previous value
%% Token from Claude fix inf loop in Matlabcode
X = zeros(N-p,p);
for i = 1:N-p
    X(i,:) = xdata(i+p-1:-1:i);
end
a = -X\x;
fprintf('T.1.b\n');
disp('Predictor coefficients a = ');
disp(a); 


%  T.1.c Prediction + error
xhat1 = -X*a;

b = [0; -flipud(a)];        
xhat2_full = filter(b,1,xdata);
xhat2 = xhat2_full(p+1:N);
figure;
plot(p+1:N,x,'LineWidth',1.2); hold on;
plot(p+1:N,xhat1,'--','LineWidth',1.2);
plot(p+1:N,xhat2,':','LineWidth',1.2);
legend('Actual','Matrix prediction','Filter prediction');
xlabel('Week'); ylabel('DJIA');
title('T.1.c Prediction Comparison');
grid on;

% error1
e1 = X*a + x;
sse1 = e1.'*e1;

% error2
e2 = x - xhat2;
sse2 = e2.'*e2;

fprintf('T.1.c\n');
fprintf('SSE (Xa + x) = %.6f\n', sse1);
fprintf('SSE (actual - predicted) = %.6f\n\n', sse2);


%  T.1.d Error vs p
p_list = 1:10;
sse_all = zeros(length(p_list),1);
for pp = p_list
    xdata_p = djia(1:N);
    x_p = xdata_p(pp+1:N);
    X_p = zeros(N-pp,pp);
    for i = 1:N-pp
        X_p(i,:) = xdata_p(i+pp-1:-1:i);
    end
    a_p = -X_p\x_p;
    e_p = X_p*a_p + x_p;
    
    sse_all(pp) = e_p.'*e_p;
end
figure;
plot(p_list,sse_all,'o-','LineWidth',1.2);
xlabel('p'); ylabel('SSE');
title('T.1.d Prediction Error vs Model Order');
grid on;

% select the best candidate of the p
%After the Graph I think P==3 or P==4 both work for this case
%% Toke from Chat GPT which point should be the best? P=3 or P=4
ratio = zeros(length(p_list),1);
ratio(1)=NaN;
for k=2:length(p_list)
    ratio(k)=abs(sse_all(k)-sse_all(k-1))/sse_all(k-1);
end
idx = find(ratio<0.01,1);
if isempty(idx)
    p_best = 3;
else
    p_best = idx;
end
fprintf('Selected p = %d\n\n', p_best);

%  T.1.e Investment (train decade)
p = p_best;
train_data = djia(1:N);
x_train = train_data(p+1:N);
X_train = zeros(N-p,p);
for i = 1:N-p
    X_train(i,:) = train_data(i+p-1:-1:i);
end
a = -X_train\x_train;
money_omniscient = initial_money;
money_bank = initial_money;
money_buyhold = initial_money;
money_predictor = initial_money;
for n = p:N-1
    actual_gain = djia(n+1)/djia(n);
    
    %the best
    money_omniscient = money_omniscient * max(actual_gain, bank_weekly);
    %Always bank
    money_bank = money_bank * bank_weekly;
    %Always DJIA
    money_buyhold = money_buyhold * actual_gain;
    
    xpast = djia(n:-1:n-p+1);
    xpred = -a.'*xpast;
    pred_gain = xpred/djia(n);
    
    if pred_gain > bank_weekly
        money_predictor = money_predictor * actual_gain;
    else
        money_predictor = money_predictor * bank_weekly;
    end
end
weeks_used = N-p;
apr_predictor = 52*((money_predictor/initial_money)^(1/weeks_used)-1);
fprintf('T.1.e\n');
fprintf('Omniscient = %.6f\n', money_omniscient);
fprintf('Bank = %.6f\n', money_bank);
fprintf('Buy-hold = %.6f\n', money_buyhold);
fprintf('Predictor = %.6f\n', money_predictor);
fprintf('Equivalent APR = %.6f\n\n', apr_predictor);

%  T.1.f Recent decade test
test_len = 520;
start_idx = Ntotal - test_len + 1;

money_omniscient = initial_money;
money_bank = initial_money;
money_buyhold = initial_money;
money_predictor = initial_money;

for n = start_idx+p-1:Ntotal-1
    
    actual_gain = djia(n+1)/djia(n);
    
    money_omniscient = money_omniscient * max(actual_gain, bank_weekly);
    money_bank = money_bank * bank_weekly;
    money_buyhold = money_buyhold * actual_gain;
    
    xpast = djia(n:-1:n-p+1);
    xpred = -a.'*xpast;
    pred_gain = xpred/djia(n);
    
    if pred_gain > bank_weekly
        money_predictor = money_predictor * actual_gain;
    else
        money_predictor = money_predictor * bank_weekly;
    end
end
apr_recent = 52*((money_predictor/initial_money)^(1/(test_len-p))-1);
fprintf('T.1.f\n');
fprintf('Predictor (recent) = %.6f\n');
fprintf('Equivalent APR = %.6f\n\n', apr_recent);

%  T.1.g Maximum possible
money_max = initial_money;

for n=1:Ntotal-1
    gain = djia(n+1)/djia(n);
    money_max = money_max * max(gain, bank_weekly);
end

fprintf('T.1.g\n');
fprintf('Max possible gain = %.6f\n\n', money_max);

%  T.1.h Frequency modeling
xdata = djia(1:N);
x = xdata(p+1:N);
X = zeros(N-p,p);
for i = 1:N-p
    X(i,:) = xdata(i+p-1:-1:i);
end
a = -X\x;
e = X*a + x;
nfft = 4096;
Xf = fftshift(fft(xdata,nfft));
w = linspace(-pi,pi,nfft);
A = [1; a];
H = freqz(1,A,nfft,'whole');
H = fftshift(H);
figure;
plot(w,abs(Xf),'LineWidth',1.2); hold on;
plot(w,abs(H),'LineWidth',1.2);
legend('|X|','|Xhat|');
title('Frequency Comparison');
grid on;
G = sqrt(sum(abs(x).^2)/sum(abs(e).^2));
H_scaled = G*H;
figure;
plot(w,abs(Xf),'LineWidth',1.2); hold on;
plot(w,abs(H_scaled),'LineWidth',1.2);
legend('|X|','|G*Xhat|');
title('Scaled Frequency');
grid on;