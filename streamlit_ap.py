import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import dask.array as da
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization, Input
import h5py
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
# from tensorflow.keras.models import load_model


# -------------------------------
# Page config
# -------------------------------


st.set_page_config(
    page_title="Drought Forecast – Tunisia",
    page_icon="🌍",
    layout="wide"
)
# -------------------------------
# University Header
# -------------------------------

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.header-row img { margin-top: 0px !important; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])

with col1:
    st.markdown('<div class="header-row">', unsafe_allow_html=True)
    st.image("assets/university_logo.png", width=40)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="header-row">', unsafe_allow_html=True)
    st.markdown("""
    <h5 style="margin-left: -85px; padding: 0;">
    Institut supérieur des arts multimédia de La Manouba
    </h5>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()



st.markdown("""
<h3 style="text-align:center;">🌍 Drought Forecasting – Tunisia</h3>
<h5 style="text-align:center; color:gray;">
ConvLSTM + Theory of Evidence (Yager Fusion)
</h5>
""", unsafe_allow_html=True)



# -------------------------------
# Load data
# -------------------------------
ds = xr.open_dataset("belief_visualization.nc",engine="netcdf4", chunks={"time": 1})

# Load dataset

# Extract belief maps (lazy)
d_map = ds["Drought_Belief"].data
n_map = ds["Normal_Belief"].data
w_map = ds["Wet_Belief"].data
u_map = ds["Uncertainty"].data

# Stack → (time, y, x, 4)
data = da.stack([d_map, n_map, w_map, u_map], axis=-1)

# Fix accidental 5D (month dimension)
if data.ndim == 5:
    data = data.mean(axis=3)

print("Final tensor shape:", data.shape)
# Expected: (299, 851, 410, 4)


# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("Controls")

    max_t = data.shape[0] - 13  # need 12 for input + 1 target

    t = st.slider(
        "Select time index",
        min_value=0,
        max_value=int(max_t),
        value=0
    )

    belief = st.radio(
        "Belief map",
        ["Drought", "Normal", "Wet", "Uncertainty"]
    )
# -------------------------------
# University Header

belief_channels = {
    "Drought": 0,
    "Normal": 1,
    "Wet": 2,
    "Uncertainty": 3
}

ch = belief_channels[belief]

# Real map (next timestep)
real_map = data[t + 12, :, :, ch].compute()
# Predicted map (mean over next 12 timesteps)

fig_real = px.imshow(
    real_map,
    color_continuous_scale="Viridis",
    aspect="auto",
    title=f"Real {belief} Belief"
)

fig_real.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_showscale=True
)

st.plotly_chart(fig_real, use_container_width=True)

seq_len = 12
rows, cols, features = 12, 851, 410, 4  # adjust if your tensor shape differs

inputs = Input(shape=(seq_len, rows, cols, features))
x = ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same', return_sequences=False)(inputs)
outputs = Conv2D(filters=4, kernel_size=(1,1), activation='sigmoid')(x)

model = Model(inputs, outputs)

# Load weights
model.load_weights("conv_lstm_final.h5")
print("✅ Model loaded successfully!")


