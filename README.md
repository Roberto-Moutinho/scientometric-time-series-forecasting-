# 📊 Scientometric Time Series Forecasting  
### Large-Scale Forecasting of Brazilian Scientific Production Using LSTM and Nonlinear Regression

---

## 🚀 Overview

This project implements a complete **time-series forecasting pipeline** to model and predict long-term trends in Brazilian scientific production using real-world large-scale academic data.

The system compares classical nonlinear regression with deep learning (LSTM) to evaluate performance under structural changes and volatility regimes.

The forecasting framework developed here is directly transferable to:

- Demand forecasting  
- Product growth modeling  
- KPI trend prediction  
- Revenue forecasting  
- User activity modeling  

---

## 🎯 Problem Statement

Scientific production evolves under the influence of:

- Public funding policies  
- Institutional strategy  
- Economic cycles  
- Global disruptions  

This project models and forecasts:

- 📄 Journal publications  
- 🔬 Research projects  
- 🎓 Academic advisories  

Across the 8 major CAPES knowledge areas in Brazil.

The goal is to:

- Model historical growth and decline patterns  
- Compare baseline and deep learning models  
- Quantitatively evaluate predictive robustness  
- Extract structural insights from national scientometric data  

---

## 📚 Dataset

### Sources

- Brazilian Lattes Platform  
- OpenAlex  
- UFABC Scientometric Database  

### Characteristics

- Multi-decade historical data (1990–2024)  
- National aggregated production  
- 8 knowledge areas  
- Millions of individual academic records processed upstream  

### Features

- Annual publication counts  
- Annual research project initiations  
- Annual academic advisories  
- Area-based aggregation  
- Structured time-indexed sequences  

---

## 🏗 Pipeline Architecture

End-to-end workflow


### Engineering Decisions

- Strict chronological splitting (no data leakage)
- Sequence windowing for LSTM training
- Feature scaling per series
- Modular model abstraction
- Deterministic seed control for reproducibility

---

## 🤖 Models Implemented

### 1️⃣ Polynomial Nonlinear Regression

- High-degree polynomial fitting
- Deterministic closed-form solution
- Interpretable long-term trend baseline

---

### 2️⃣ LSTM (Long Short-Term Memory)

Deep recurrent neural network designed to capture:

- Long-range temporal dependencies  
- Nonlinear growth patterns  
- Structural regime shifts  

Configuration:

- Sliding input windows  
- Dense output layer  
- ReLU activation  
- Adam optimizer  
- Early stopping  

---

## 📈 Evaluation Strategy

Proper time-series validation:

- Train: Historical segment  
- Validation: Recent years  
- Test: Final unseen segment  

### Metrics

- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- MAPE (Mean Absolute Percentage Error)  

Evaluation performed across:

- Knowledge areas  
- Model classes  
- Volatility regimes  

---

## 🔎 Key Findings

- Polynomial regression performs competitively in stable growth regimes.
- LSTM significantly outperforms regression during:
  - High volatility periods  
  - Post-peak structural decline  
  - Recovery phases  
- Most knowledge areas peaked between 2010–2020.
- Post-2020 decline observed across multiple domains.
- Forecasts suggest partial recovery without full return to historical maxima.

---

## 💡 Technical Highlights

- Real-world large-scale dataset  
- Deep learning applied to structured time-series data  
- Baseline vs neural model comparison  
- Quantitative performance evaluation  
- Reproducible ML pipeline  
- Cross-domain comparative analysis  

---



