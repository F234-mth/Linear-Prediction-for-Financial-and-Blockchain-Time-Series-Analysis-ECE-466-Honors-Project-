clear; clc; close all;
load('eth_2019.mat');
dates = block_difficulty(:,1);
difficulty = block_difficulty(:,2);
date_strings = datestr(dates);
%% T.2.a
%  Training: July 30, 2015 to Dec 31, 2015
%  Prediction: Jan 1, 2016 to Jun 30, 2016
train_start_a = datenum('30-Jul-2015');% Compute without using string 
train_end_a   = datenum('31-Dec-2015');
test_start_a  = datenum('01-Jan-2016');
test_end_a    = datenum('30-Jun-2016');
idx_train_a = find(dates >= train_start_a & dates <= train_end_a);%using the 7.30.2015 -- 12.31.2016 for training index
idx_test_a  = find(dates >= test_start_a  & dates <= test_end_a);%using 1.1.2016 -- 6.30.2016 for test index
train_data_a = difficulty(idx_train_a);
test_data_a  = difficulty(idx_test_a);
test_dates_a = dates(idx_test_a);
%T.2.a(i): choose p = 2, design predictor, plot predicted and real difficulties
p = 2;
Ntrain = length(train_data_a);
x = train_data_a(p+1:Ntrain);
X = zeros(Ntrain-p,p);

for i = 1:Ntrain-p
    X(i,:) = train_data_a(i+p-1:-1:i);
end

a = -X\x;

% Recursive multi-step prediction:
% use last p samples from training data first,
% then keep appending predictions
pred_a_p2 = zeros(length(test_data_a),1);
history = train_data_a;

for n = 1:length(test_data_a)
    xpast = history(end:-1:end-p+1);
    pred_a_p2(n) = -a.' * xpast;
    history = [history; pred_a_p2(n)];
end

figure;
plot(test_dates_a,test_data_a,'LineWidth',1.2); hold on;
plot(test_dates_a,pred_a_p2,'--','LineWidth',1.2);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.a(i): Ethereum Difficulty Prediction, p = 2');
legend('Actual','Predicted','Location','best');
grid on;

%one-step-ahead method doing a(i)
%token from Chat GPT "how to maake the prediction more accurate by using-
%-linaer regression "
train_start_a = datenum('30-Jul-2015');
train_end_a   = datenum('31-Dec-2015');
test_start_a  = datenum('01-Jan-2016');
test_end_a    = datenum('30-Jun-2016');
idx_train_a = find(dates >= train_start_a & dates <= train_end_a);
idx_test_a  = find(dates >= test_start_a  & dates <= test_end_a);
train_data_a = difficulty(idx_train_a);
test_data_a  = difficulty(idx_test_a);
test_dates_a = dates(idx_test_a);
p = 2;
Ntrain = length(train_data_a);
x = train_data_a(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_data_a(i+p-1:-1:i);
end
a = -X\x;
full_data = [train_data_a; test_data_a];
pred_a_p2 = zeros(length(test_data_a),1);
for n = 1:length(test_data_a)
    current_index = length(train_data_a) + n;
    xpast = full_data(current_index-1:-1:current_index-p);
    pred_a_p2(n) = -a.' * xpast;
end
figure;
plot(test_dates_a,test_data_a,'LineWidth',1.2); hold on;
plot(test_dates_a,pred_a_p2,'--','LineWidth',1.2);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.a(i): Ethereum Difficulty Prediction, p = 2');
legend('Actual','Predicted','Location','best');
grid on;



% T.2.a(ii): vary p from 2 to 50 in steps of 4
% compute least squares training error versus p
p_list = 2:4:50;
sse_train = zeros(length(p_list),1);
for k = 1:length(p_list)
    p = p_list(k);
    Ntrain = length(train_data_a);
    x = train_data_a(p+1:Ntrain);
    X = zeros(Ntrain-p,p);
    for i = 1:Ntrain-p
        X(i,:) = train_data_a(i+p-1:-1:i);
    end
    a = -X\x;
    e = X*a + x;
    sse_train(k) = e.'*e;
end
figure;
plot(p_list,sse_train,'o-','LineWidth',1.2);
xlabel('Predictor Order p');
ylabel('Least Squares Error');
title('T.2.a(ii): Training Least Squares Error vs p');
grid on;
%  T.2.a(iii): average predicted error for p from 2 to 50
%  Average predicted error equation:(1/N) * sum( (d_hat(n) - d(n))^2 )
%  Here prediction is done on Jan 1, 2016 to Jun 30, 2016
avg_pred_error_a = zeros(length(p_list),1);
for k = 1:length(p_list)
    p = p_list(k);
    Ntrain = length(train_data_a);
    x = train_data_a(p+1:Ntrain);
    X = zeros(Ntrain-p,p);
    for i = 1:Ntrain-p
        X(i,:) = train_data_a(i+p-1:-1:i);
    end
    a = -X\x;
    pred = zeros(length(test_data_a),1);
    history = train_data_a;
    for n = 1:length(test_data_a)
        xpast = history(end:-1:end-p+1);
        pred(n) = -a.' * xpast;
        history = [history; pred(n)];
    end
    avg_pred_error_a(k) = mean((pred - test_data_a).^2);
end

figure;
plot(p_list,avg_pred_error_a,'s-','LineWidth',1.2);
xlabel('Predictor Order p');
ylabel('Average Predicted Error');
title('T.2.a(iii): Average Prediction Error vs p');
grid on;


%% T.2.a(iii): Average predicted error vs p using one-step-ahead prediction
train_start_a = datenum('30-Jul-2015');
train_end_a   = datenum('31-Dec-2015');
test_start_a  = datenum('01-Jan-2016');
test_end_a    = datenum('30-Jun-2016');
idx_train_a = find(dates >= train_start_a & dates <= train_end_a);
idx_test_a  = find(dates >= test_start_a  & dates <= test_end_a);
train_data_a = difficulty(idx_train_a);
test_data_a  = difficulty(idx_test_a);
p_list = 2:4:50;
avg_pred_error_a = zeros(length(p_list),1);
full_data = [train_data_a; test_data_a];
for k = 1:length(p_list)
    p = p_list(k);
    Ntrain = length(train_data_a);
    x = train_data_a(p+1:Ntrain);
    X = zeros(Ntrain-p,p);
    for i = 1:Ntrain-p
        X(i,:) = train_data_a(i+p-1:-1:i);
    end
    a = -X\x;
    pred = zeros(length(test_data_a),1);
    for n = 1:length(test_data_a)
        current_index = length(train_data_a) + n;
        xpast = full_data(current_index-1:-1:current_index-p);
        pred(n) = -a.' * xpast;
    end
    avg_pred_error_a(k) = mean((pred - test_data_a).^2);
end
figure;
plot(p_list,avg_pred_error_a,'o-','LineWidth',1.2);
xlabel('Predictor Order p');
ylabel('Average Predicted Error');
title('T.2.a(iii): Average Prediction Error vs p');
grid on;
%  T.2.b
%  Training: Jan 1, 2016 to Dec 31, 2016
%  Predictor order p = 2
%  (a) Predict Jan 1, 2017 to Dec 31, 2017
%  (b) Predict Jan 1, 2018 to Dec 31, 2018
%  (c) Compare average predicted errors
%% here is using the recursive function 
train_start_b = datenum('01-Jan-2016');
train_end_b   = datenum('31-Dec-2016');

test1_start_b = datenum('01-Jan-2017');
test1_end_b   = datenum('31-Dec-2017');

test2_start_b = datenum('01-Jan-2018');
test2_end_b   = datenum('31-Dec-2018');

idx_train_b = find(dates >= train_start_b & dates <= train_end_b);
idx_test1_b = find(dates >= test1_start_b & dates <= test1_end_b);
idx_test2_b = find(dates >= test2_start_b & dates <= test2_end_b);

train_data_b = difficulty(idx_train_b);
test1_data_b = difficulty(idx_test1_b);
test2_data_b = difficulty(idx_test2_b);

test1_dates_b = dates(idx_test1_b);
test2_dates_b = dates(idx_test2_b);

p = 2;
Ntrain = length(train_data_b);
x = train_data_b(p+1:Ntrain);
X = zeros(Ntrain-p,p);

for i = 1:Ntrain-p
    X(i,:) = train_data_b(i+p-1:-1:i);
end
a = -X\x;
%  T.2.b(a): Predict all of 2017
pred_2017 = zeros(length(test1_data_b),1);
history = train_data_b;
for n = 1:length(test1_data_b)
    xpast = history(end:-1:end-p+1);
    pred_2017(n) = -a.' * xpast;
    history = [history; pred_2017(n)];
end
figure;
plot(test1_dates_b,test1_data_b,'LineWidth',1.2); hold on;
plot(test1_dates_b,pred_2017,'--','LineWidth',1.2);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.b(a): Predicting 2017 Difficulty, p = 2');
legend('Actual','Predicted','Location','best');
grid on;
%  T.2.b(b): Predict all of 2018
pred_2018 = zeros(length(test2_data_b),1);
history = train_data_b;
for n = 1:length(test2_data_b)
    xpast = history(end:-1:end-p+1);
    pred_2018(n) = -a.' * xpast;
    history = [history; pred_2018(n)];
end
figure;
plot(test2_dates_b,test2_data_b,'LineWidth',1.2); hold on;
plot(test2_dates_b,pred_2018,'--','LineWidth',1.2);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.b(b): Predicting 2018 Difficulty, p = 2');
legend('Actual','Predicted','Location','best');
grid on;
%  T.2.b(c): Average predicted errors for 2017 and 2018
avg_error_2017 = mean((pred_2017 - test1_data_b).^2);
avg_error_2018 = mean((pred_2018 - test2_data_b).^2);
fprintf('T.2.b(c)\n');
fprintf('Average predicted error for 2017 = %.6f\n', avg_error_2017);
fprintf('Average predicted error for 2018 = %.6f\n\n', avg_error_2018);

%% here is using the linear regression one step method
train_start_b = datenum('01-Jan-2016');
train_end_b   = datenum('31-Dec-2016');
test1_start_b = datenum('01-Jan-2017');
test1_end_b   = datenum('31-Dec-2017');
test2_start_b = datenum('01-Jan-2018');
test2_end_b   = datenum('31-Dec-2018');
idx_train_b = find(dates >= train_start_b & dates <= train_end_b);
idx_test1_b = find(dates >= test1_start_b & dates <= test1_end_b);
idx_test2_b = find(dates >= test2_start_b & dates <= test2_end_b);
train_data_b = difficulty(idx_train_b);
test1_data_b = difficulty(idx_test1_b);
test2_data_b = difficulty(idx_test2_b);
test1_dates_b = dates(idx_test1_b);
test2_dates_b = dates(idx_test2_b);
p = 2;
Ntrain = length(train_data_b);
x = train_data_b(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_data_b(i+p-1:-1:i);
end
a = -X\x;
%% T.2.b(a): One-step-ahead prediction for 2017
full_data_2017 = [train_data_b; test1_data_b];
pred_2017 = zeros(length(test1_data_b),1);
for n = 1:length(test1_data_b)
    current_index = length(train_data_b) + n;
    xpast = full_data_2017(current_index-1:-1:current_index-p);
    pred_2017(n) = -a.' * xpast;
end
figure;
plot(test1_dates_b,test1_data_b,'LineWidth',1.2); hold on;
plot(test1_dates_b,pred_2017,'--','LineWidth',1.2);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.b(a): Predicting 2017 Difficulty, One-Step, p = 2');
legend('Actual','Predicted','Location','best');
grid on;
%% T.2.b(b): One-step-ahead prediction for 2018
full_data_2018 = [train_data_b; test2_data_b];
pred_2018 = zeros(length(test2_data_b),1);
for n = 1:length(test2_data_b)
    current_index = length(train_data_b) + n;
    xpast = full_data_2018(current_index-1:-1:current_index-p);
    pred_2018(n) = -a.' * xpast;
end
figure;
plot(test2_dates_b,test2_data_b,'LineWidth',1.2); hold on;
plot(test2_dates_b,pred_2018,'--','LineWidth',1.2);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.b(b): Predicting 2018 Difficulty, One-Step, p = 2');
legend('Actual','Predicted','Location','best');
grid on;
%% T.2.b(c): Average one-step prediction errors
avg_error_2017 = mean((pred_2017 - test1_data_b).^2);
avg_error_2018 = mean((pred_2018 - test2_data_b).^2);
fprintf('T.2.b(c)\n');
fprintf('Average one-step predicted error for 2017 = %.6f\n', avg_error_2017);
fprintf('Average one-step predicted error for 2018 = %.6f\n\n', avg_error_2018);
%% I am also think the FxLMS can be applied to this case or not? 
%% but we do not have a secondary path here, LMS can converge error to zero very quick 
%% but it is not always zero it osilate between zero in a range








%  T.2.c recursive method
%  Predict Jan 1, 2018 to Jun 30, 2018 using previous:
%   (a) one year (365 days)
%   (b) 6 months (180 days)
%   (c) one month (30 days)
%  Use predictor order p = 2 unless your instructor says otherwise
pred_start_c = datenum('01-Jan-2018');
pred_end_c   = datenum('30-Jun-2018');
idx_pred_c = find(dates >= pred_start_c & dates <= pred_end_c);
pred_dates_c = dates(idx_pred_c);
actual_c = difficulty(idx_pred_c);

p = 2;
%  Case 1: use previous 365 days of data
train_end_c = pred_start_c - 1;
train_start_365 = train_end_c - 364;
idx_train_365 = find(dates >= train_start_365 & dates <= train_end_c);
train_365 = difficulty(idx_train_365);
Ntrain = length(train_365);
x = train_365(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_365(i+p-1:-1:i);
end
a_365 = -X\x;
pred_365 = zeros(length(actual_c),1);
history = train_365;
for n = 1:length(actual_c)
    xpast = history(end:-1:end-p+1);
    pred_365(n) = -a_365.' * xpast;
    history = [history; pred_365(n)];
end
%  Case 2: use previous 180 days of data
train_start_180 = train_end_c - 179;
idx_train_180 = find(dates >= train_start_180 & dates <= train_end_c);
train_180 = difficulty(idx_train_180);
Ntrain = length(train_180);
x = train_180(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_180(i+p-1:-1:i);
end
a_180 = -X\x;
pred_180 = zeros(length(actual_c),1);
history = train_180;
for n = 1:length(actual_c)
    xpast = history(end:-1:end-p+1);
    pred_180(n) = -a_180.' * xpast;
    history = [history; pred_180(n)];
end
%  Case 3: use previous 30 days of data
train_start_30 = train_end_c - 29;
idx_train_30 = find(dates >= train_start_30 & dates <= train_end_c);
train_30 = difficulty(idx_train_30);
Ntrain = length(train_30);
x = train_30(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_30(i+p-1:-1:i);
end
a_30 = -X\x;
pred_30 = zeros(length(actual_c),1);
history = train_30;
for n = 1:length(actual_c)
    xpast = history(end:-1:end-p+1);
    pred_30(n) = -a_30.' * xpast;
    history = [history; pred_30(n)];
end
figure;
plot(pred_dates_c,actual_c,'k','LineWidth',1.4); hold on;
plot(pred_dates_c,pred_365,'--','LineWidth',1.2);
plot(pred_dates_c,pred_180,'-.','LineWidth',1.2);
plot(pred_dates_c,pred_30,':','LineWidth',1.5);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.c: Prediction of Ethereum Difficulty (Jan 1, 2018 to Jun 30, 2018)');
legend('Actual','Train with 365 days','Train with 180 days','Train with 30 days','Location','best');
grid on;
avg_error_365 = mean((pred_365 - actual_c).^2);
avg_error_180 = mean((pred_180 - actual_c).^2);
avg_error_30  = mean((pred_30  - actual_c).^2);
fprintf('T.2.c\n');
fprintf('Average error using previous 365 days = %.6f\n', avg_error_365);
fprintf('Average error using previous 180 days = %.6f\n', avg_error_180);
fprintf('Average error using previous 30 days  = %.6f\n', avg_error_30);



%% T.2.c One-Step-Ahead Prediction
% Predict Jan 1, 2018 to Jun 30, 2018 using previous:
% (a) 365 days
% (b) 180 days
% (c) 30 days

pred_start_c = datenum('01-Jan-2018');
pred_end_c   = datenum('30-Jun-2018');

idx_pred_c = find(dates >= pred_start_c & dates <= pred_end_c);
pred_dates_c = dates(idx_pred_c);
actual_c = difficulty(idx_pred_c);

p = 2;

%% Case 1: use previous 365 days of data
train_end_c = pred_start_c - 1;
train_start_365 = train_end_c - 364;
idx_train_365 = find(dates >= train_start_365 & dates <= train_end_c);
train_365 = difficulty(idx_train_365);
Ntrain = length(train_365);
x = train_365(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_365(i+p-1:-1:i);
end
a_365 = -X\x;
full_data_365 = [train_365; actual_c];
pred_365 = zeros(length(actual_c),1);
for n = 1:length(actual_c)
    current_index = length(train_365) + n;
    xpast = full_data_365(current_index-1:-1:current_index-p);
    pred_365(n) = -a_365.' * xpast;
end
%% Case 2: use previous 180 days of data
train_start_180 = train_end_c - 179;
idx_train_180 = find(dates >= train_start_180 & dates <= train_end_c);
train_180 = difficulty(idx_train_180);
Ntrain = length(train_180);
x = train_180(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_180(i+p-1:-1:i);
end
a_180 = -X\x;
full_data_180 = [train_180; actual_c];
pred_180 = zeros(length(actual_c),1);
for n = 1:length(actual_c)
    current_index = length(train_180) + n;
    xpast = full_data_180(current_index-1:-1:current_index-p);
    pred_180(n) = -a_180.' * xpast;
end
%% Case 3: use previous 30 days of data
train_start_30 = train_end_c - 29;
idx_train_30 = find(dates >= train_start_30 & dates <= train_end_c);
train_30 = difficulty(idx_train_30);
Ntrain = length(train_30);
x = train_30(p+1:Ntrain);
X = zeros(Ntrain-p,p);
for i = 1:Ntrain-p
    X(i,:) = train_30(i+p-1:-1:i);
end
a_30 = -X\x;
full_data_30 = [train_30; actual_c];
pred_30 = zeros(length(actual_c),1);
for n = 1:length(actual_c)
    current_index = length(train_30) + n;
    xpast = full_data_30(current_index-1:-1:current_index-p);
    pred_30(n) = -a_30.' * xpast;
end
figure;
plot(pred_dates_c,actual_c,'k','LineWidth',1.4); hold on;
plot(pred_dates_c,pred_365,'--','LineWidth',1.2);
plot(pred_dates_c,pred_180,'-.','LineWidth',1.2);
plot(pred_dates_c,pred_30,':','LineWidth',1.5);
datetick('x','mm/dd/yyyy','keeplimits');
xlabel('Date');
ylabel('Difficulty');
title('T.2.c: One-Step-Ahead Prediction of Ethereum Difficulty');
legend('Actual','Train with 365 days','Train with 180 days','Train with 30 days','Location','best');
grid on;

%% Compute average errors
avg_error_365 = mean((pred_365 - actual_c).^2);
avg_error_180 = mean((pred_180 - actual_c).^2);
avg_error_30  = mean((pred_30  - actual_c).^2);

fprintf('T.2.c One-Step-Ahead\n');
fprintf('Average error using previous 365 days = %.6f\n', avg_error_365);
fprintf('Average error using previous 180 days = %.6f\n', avg_error_180);
fprintf('Average error using previous 30 days  = %.6f\n', avg_error_30);