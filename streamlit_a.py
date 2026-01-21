import streamlit as st
import xarray as xr
import numpy as np
import dask.array as da
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Drought Forecast – Tunisia",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# University Header
# -------------------------------
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/university_logo.png", width=50)
with col2:
    st.markdown("""
    <h5 style="margin-top:0;">Institut supérieur des arts multimédia de La Manouba</h5>
    <p>Prof. A. Name | Prof. B. Name | Prof. C. Name</p>
    """, unsafe_allow_html=True)
st.divider()

st.markdown("""
<h3 style="text-align:center;">🌍 Drought Forecasting – Tunisia</h3>
<h5 style="text-align:center; color:gray;">ConvLSTM + Theory of Evidence (Yager Fusion)</h5>
""", unsafe_allow_html=True)

# -------------------------------
# Load Real Data
# -------------------------------
ds = xr.open_dataset("belief_visualization.nc", chunks={"time": 1})

# Extract Dask arrays
d_map = ds["Drought_Belief"].data
n_map = ds["Normal_Belief"].data
w_map = ds["Wet_Belief"].data
u_map = ds["Uncertainty"].data

# Stack → (time, y, x, 4)
data = da.stack([d_map, n_map, w_map, u_map], axis=-1)

# Collapse month if needed
if data.ndim == 5:
    data = data.mean(axis=3)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Controls")
time_idx = st.sidebar.slider("Select time index", 0, data.shape[0]-12, 0)
map_type = st.sidebar.radio("Select Map Type", ["Drought", "Normal", "Wet", "Uncertainty"])

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_conv_lstm_model(path="conv_lstm_final.h5"):
    model = load_model(path)
    return model

model = load_conv_lstm_model()

# -------------------------------
# Prepare Input Sequence
# -------------------------------
SEQ_LEN = 12
X_seq = data[time_idx:time_idx+SEQ_LEN].compute()  # Compute Dask array
X_seq = np.expand_dims(X_seq, axis=0).astype(np.float32)  # Add batch

# -------------------------------
# Predict
# -------------------------------
y_pred = model.predict(X_seq)[0]  # Remove batch dim

# -------------------------------
# Select Real Map Slice
# -------------------------------
real_map = data[time_idx+SEQ_LEN].compute()  # Shape: (y, x, 4)

# -------------------------------
# Mapping dictionary
# -------------------------------
channel_dict = {
    "Drought": 0,
    "Normal": 1,
    "Wet": 2,
    "Uncertainty": 3
}

idx = channel_dict[map_type]

# -------------------------------
# Normalize for plotting
# -------------------------------
def normalize_img(img):
    img = np.array(img)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)  # avoid div by 0
    return img_norm

real_norm = normalize_img(real_map[..., idx])
pred_norm = normalize_img(y_pred[..., idx])

# -------------------------------
# Plot side-by-side
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### Real {map_type} Map")
    fig1 = px.imshow(real_norm, color_continuous_scale="inferno")
    fig1.update_xaxes(showticklabels=False)
    fig1.update_yaxes(showticklabels=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown(f"### Predicted {map_type} Map")
    fig2 = px.imshow(pred_norm, color_continuous_scale="inferno")
    fig2.update_xaxes(showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Master Project – Drought Forecasting Tunisia")
