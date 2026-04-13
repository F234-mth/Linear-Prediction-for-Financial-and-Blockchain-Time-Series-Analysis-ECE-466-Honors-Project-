Linear Prediction for Financial and Blockchain Time Series Analysis
Overview
This project explores linear prediction techniques for modeling and forecasting time-series data, with applications to:
	•	Dow Jones Industrial Average (DJIA)
	•	⛓️ Ethereum blockchain difficulty

The project covers:
	•	Least-squares linear predictor design
	•	Time-domain error minimization
	•	Frequency-domain interpretation (Parseval’s relation)
	•	Investment strategy simulation
	•	Comparison of prediction methods (recursive vs one-step)
	•	Extension to adaptive methods (LMS / RLS)


 Key Concepts

 Linear Prediction Model

The signal is modeled as a linear combination of past samples:

\hat{x}[n] = -\sum_{k=1}^{p} a_k x[n-k]
	•	a_k: predictor coefficients
	•	p: model order
	•	\hat{x}[n]: predicted value

This is equivalent to a time-series linear regression model.

⸻

Prediction Error

e[n] = x[n] - \hat{x}[n]

Total squared error:

E = \sum_{n=1}^{N} |e[n]|^2

The objective is to minimize E → leads to a least-squares solution.

⸻

Matrix Formulation

X a \approx -x

Solution in MATLAB:
  a = -X\x;
This computes the optimal coefficients minimizing:

\|Xa + x\|^2

⸻

Frequency-Domain Interpretation

Using Parseval’s relation:

\frac{1}{2\pi} \int_{-\pi}^{\pi}
\left|
\frac{X(e^{j\omega})}{\hat{X}(e^{j\omega})}
\right|^2 d\omega

Minimizing time-domain error ≈ matching spectral characteristics
Linear prediction can also be viewed as spectral modeling


 Prediction Methods

One-Step-Ahead Prediction
	•	Uses true past values
	•	No error accumulation
	•	More accurate for short-term prediction

 Recursive Prediction
	•	Uses predicted values as inputs
	•	Errors accumulate over time
	•	Simulates real forecasting scenarios


Experiments

1. DJIA Analysis
	•	Linear vs semi-log plots
	•	Investment strategies:
	•	Bank (3% APR)
	•	Buy-and-hold
	•	Predictor-based strategy
	•	Omniscient upper bound

2. Ethereum Difficulty Prediction
	•	Training vs testing periods
	•	Model order selection
	•	Comparison of:
	•	One-step vs recursive prediction
	•	Different training window sizes (365 / 180 / 30 days)


Key Findings
	•	Linear prediction works well for short-term forecasting
	•	Recursive prediction suffers from error accumulation
	•	Larger model order p → risk of overfitting
	•	Financial and blockchain data are non-stationary
	•	Training window tradeoff:
	•	Long window → stability
	•	Short window → adaptability


Linear Regression Perspective

Linear prediction is equivalent to:

y = w^T x

where:
	•	y = x[n]
	•	x = [x[n-1], x[n-2], \dots, x[n-p]]
	•	w = -a

This is a time-shifted linear regression model

Adaptive Extension (RLS)

To handle non-stationary data, adaptive methods like Recursive Least Squares (RLS) can be used.

RLS Update Equations

k[n] = \frac{P[n-1]x[n]}{\lambda + x^T[n]P[n-1]x[n]}

a[n] = a[n-1] + k[n] e[n]

P[n] = \frac{1}{\lambda} \left( P[n-1] - k[n] x^T[n] P[n-1] \right)
	•	\lambda: forgetting factor
	•	Allows model to adapt over time

Much better for:
	•	financial data
	•	blockchain difficulty


Project Structure
.
├── data/
│   ├── djia_2019.mat
│   └── eth_2019.mat
├── matlab/
│   ├── Honors466.m
│   ├── honors466p2.m
├── figures/
│   ├── plots_djia/
│   ├── plots_eth/
│   └── frequency_analysis/
Figures shows in Report
├── report/
│   └── final_report.pdf
└── README.md

📌 Applications
	•	Financial forecasting
	•	Cryptocurrency modeling
	•	Signal processing
	•	Adaptive filtering
	•	Spectral estimation



⚠️ Limitations
	•	Assumes linearity
	•	Sensitive to non-stationarity
	•	Recursive prediction unstable for long horizons
	•	Cannot capture nonlinear market behavior



📚 References
	•	Proakis & Manolakis, Digital Signal Processing
	•	Oppenheim & Schafer, Discrete-Time Signal Processing
	•	J. Makhoul, Linear Prediction: A Tutorial Review



