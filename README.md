# Linear Prediction for Financial and Blockchain Time Series Analysis

## Overview

This project explores **linear prediction** techniques for modeling and forecasting time-series data, with applications to:

- **Dow Jones Industrial Average (DJIA)**
- **Ethereum blockchain difficulty**

The project covers:

- Least-squares linear predictor design
- Time-domain error minimization
- Frequency-domain interpretation via Parseval’s relation
- Investment strategy simulation
- Comparison of prediction methods (recursive vs. one-step-ahead)
- Extension to adaptive methods such as **LMS** and **RLS**

---

## Key Concepts

### Linear Prediction Model

The signal is modeled as a linear combination of past samples:

$$
\hat{x}[n] = -\sum_{k=1}^{p} a_k x[n-k]
$$

where:

- $a_k$ = predictor coefficients
- $p$ = model order
- $\hat{x}[n]$ = predicted value

This is equivalent to a **time-series linear regression model**, where past samples are used as input features to predict the current sample.

---

### Prediction Error

The prediction error is defined as:

$$
e[n] = x[n] - \hat{x}[n]
$$

The total squared error is:

$$
E = \sum_{n=1}^{N} |e[n]|^2
$$

The objective is to minimize $E$, which leads to a **least-squares solution**.

---

### Matrix Formulation

The linear prediction problem can be written in matrix form as:

$$
Xa \approx -x
$$

or equivalently,

$$
-Xa + e = x
$$

In MATLAB, the predictor coefficients are obtained by:

```matlab
a = -X\x;
```


## 🔹 Frequency-Domain Interpretation

Using Parseval’s relation:

$$
\frac{1}{2\pi}\int_{-\pi}^{\pi}
\left|
\frac{X(e^{j\omega})}{\hat{X}(e^{j\omega})}
\right|^2 d\omega
$$

Minimizing the time-domain prediction error is equivalent to matching the spectral characteristics of the signal in the frequency domain.

Linear prediction can therefore also be interpreted as a form of **spectral modeling**.

---

## ⚙️ Prediction Methods

### 🔵 One-Step-Ahead Prediction

- Uses true past values
- No error accumulation
- More accurate for short-term prediction

###  Recursive Prediction

- Uses predicted values as inputs
- Errors accumulate over time
- Simulates real forecasting scenarios

---

##  Experiments

### 1. DJIA Analysis

- Linear vs. semi-log plots
- Investment strategies:
  - Bank (3% APR)
  - Buy-and-hold
  - Predictor-based strategy
  - Omniscient upper bound

### 2. Ethereum Difficulty Prediction

- Training/testing split
- Model order selection
- Comparison of:
  - One-step vs. recursive prediction
  - Training windows (365 / 180 / 30 days)

---

##  Key Findings

- Linear prediction works well for short-term forecasting
- Recursive prediction suffers from error accumulation
- Larger model order $p$ may lead to overfitting
- Financial and blockchain data are non-stationary
- Training window tradeoff:
  - Long window $\rightarrow$ stability
  - Short window $\rightarrow$ adaptability

---

## Linear Regression Perspective

Linear prediction can be written as:

$$
y = w^T x
$$

where:

- $y = x[n]$
- $x = [x[n-1], x[n-2], \dots, x[n-p]]^T$
- $w = -a$

This shows that linear prediction is a **time-shifted linear regression model**.

---

##  Adaptive Extension (RLS)

To handle non-stationary data, **Recursive Least Squares (RLS)** can be used.

### RLS Update Equations

$$
k[n] = \frac{P[n-1]x[n]}{\lambda + x^T[n]P[n-1]x[n]}
$$

$$
a[n] = a[n-1] + k[n] e[n]
$$

$$
P[n] = \frac{1}{\lambda}\left(P[n-1] - k[n]x^T[n]P[n-1]\right)
$$

where:

- $\lambda$ = forgetting factor
- $P[n]$ = inverse correlation estimate
- $e[n]$ = prediction error

RLS is more suitable for:

- Financial time series
- Blockchain difficulty prediction

---

##  Project Structure

```text
.
├── data/
│   ├── djia_2019.mat
│   └── eth_2019.mat
├── matlab/
│   ├── Honors466.m
│   └── honors466p2.m
├── figures/
│   ├── plots_djia/
│   ├── plots_eth/
│   └── frequency_analysis/
├── report/
│   └── final_report.pdf
└── README.md






