import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Muat model dan scaler
try:
    model = joblib.load('model_akhir.pkl')
    scaler = joblib.load('scaler.pkl')

