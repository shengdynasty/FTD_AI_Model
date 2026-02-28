# FTD AI Cell-Level Failure Prediction Model

## Overview

This project develops a machine learning model to predict early neuronal failure in a Frontotemporal Dementia (FTD) cellular system.

Using early (0–24 hour) cell-level features, the model predicts whether a neuron will fail (e.g., protein aggregation, synaptic collapse, or death) within 7 days.

The goal is to identify early biological signals that drive degeneration and understand how cellular interactions contribute to disease progression.

---

## Biological Motivation

Frontotemporal Dementia (FTD) is a neurodegenerative disorder affecting frontal and temporal brain networks. 

At the cellular level, disease progression involves:

- Protein aggregation (Tau / TDP-43)
- Synaptic dysfunction
- Neuron–glia interactions
- Network collapse
- Neuronal death

We hypothesize that early measurable cellular features contain predictive information about later failure events.

This project builds an interpretable AI model to test that hypothesis.

---

## Model Overview

The pipeline:

Early 24-hour cell features  
→ XGBoost classifier  
→ Predict failure risk  
→ SHAP interpretability  

Key features include:

- Neurite growth rate  
- Calcium activity rate  
- Aggregation intensity slope  
- Microglia contact duration  

Grouped cross-validation is performed by donor to prevent data leakage and simulate real biological variability.

---

## Repository Structure
