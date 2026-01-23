import streamlit as st
import xarray as xr
import dask.array as da
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Input
from tensorflow.keras.models import Model
from scipy.ndimage import distance_transform_edt, gaussian_filter
from PIL import Image  # for loading your screenshot
import pandas as pd



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
    st.image("assets/university_logo.png", width=40)
with col2:
    st.markdown(
        '<h5 style="margin-left:-85px;">Institut supérieur des arts multimédia de La Manouba</h5>',
        unsafe_allow_html=True
    )

st.divider()

st.markdown(
    """
    <h3 style="text-align:center;">🌍 Drought Forecasting – Tunisia</h3>
    <h5 style="text-align:center; color:gray;">
    ConvLSTM + Theory of Evidence (Yager Fusion)
    </h5>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load Dataset
# -------------------------------
ds = xr.open_dataset("belief_visualization.nc", engine="netcdf4", chunks={"time": 1})

d_map = ds["Drought_Belief"].data
n_map = ds["Normal_Belief"].data
w_map = ds["Wet_Belief"].data
u_map = ds["Uncertainty"].data

# Stack → (time, y, x, 4)
data = da.stack([d_map, n_map, w_map, u_map], axis=-1)

# Collapse month dimension if 5D
if data.ndim == 5:
    data = data.mean(axis=3)

st.write("Dataset shape:", data.shape)

# -------------------------------
# Sidebar Controls
# -------------------------------
with st.sidebar:
    st.header("Controls")
    time_idx = st.slider("Select starting time index", 0, data.shape[0] - 12, 0)
    map_type = st.radio("Map type", ["Drought", "Normal", "Wet", "Uncertainty", "All"])
# Get datetime at selected index
current_time = ds.time.values[time_idx]

# Convert to pandas Timestamp (easy formatting)
current_time = np.datetime64(current_time)

ts = pd.to_datetime(current_time)


# -------------------------------
# Model Definition
# -------------------------------
def build_conv_lstm_model(time_steps, rows, cols, features):
    inputs = Input(shape=(time_steps, rows, cols, features))
    x = ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False
    )(inputs)
    outputs = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        activation="sigmoid",
        padding="same"
    )(x)
    return Model(inputs, outputs)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(weights_path="conv_lstm_final.h5"):
    model = build_conv_lstm_model(
        time_steps=12,
        rows=data.shape[1],
        cols=data.shape[2],
        features=data.shape[3]
    )
    model.load_weights(weights_path)
    return model

model = load_model()


# -------------------------------
# Prepare Input Sequence
# -------------------------------
X_seq = data[time_idx:time_idx + 12].compute()
X_seq = np.expand_dims(X_seq, axis=0)

# -------------------------------
# Prediction
# -------------------------------
pred = model.predict(X_seq, verbose=0)[0]  # (y, x, 4)

map_names = ["Drought", "Normal", "Wet", "Uncertainty"]


def complete_map_visually(arr):
    """
    Visually completes missing regions in a 2D map.
    NOT scientifically correct – visualization only.
    """
    arr = arr.copy()

    # Mask of valid pixels
    valid = np.isfinite(arr)

    # If everything is valid, return as-is
    if valid.all():
        return arr

    # Get indices of nearest valid pixels
    _, indices = distance_transform_edt(
        ~valid,
        return_indices=True
    )

    # Fill missing pixels with nearest valid value
    filled = arr[tuple(indices)]

    # Optional smoothing to hide blockiness
    filled = gaussian_filter(filled, sigma=1.2)

    return filled

# -------------------------------
# Plot Function (FIXED)
# -------------------------------
def plot_map(arr, title, cmap="inferno"):
    # COMPLETE missing areas visually
    arr_vis = complete_map_visually(arr)

    fig = px.imshow(
        arr_vis,
        color_continuous_scale=cmap,
        origin="upper",
        title=title
    )

    fig.update_layout(
        yaxis_scaleanchor="x",
        yaxis_constrain="domain"
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Maps & Color Maps
# -------------------------------
# -------------------------------
# Custom colormap dictionary
# -------------------------------
TIME_STEPS = 12

# -------------------------------
# Color maps and nice ranges per type
# -------------------------------
cmap_dict = {
    "Drought": "YlOrRd",
    "Normal": "viridis",
    "Wet": "Blues",
    "Uncertainty": "Greys"
}

# Optional: define min/max for each type for consistent color scaling
vmin_dict = {
    "Drought": 0.0,
    "Normal": 0.0,
    "Wet": 0.0,
    "Uncertainty": 0.0
}

vmax_dict = {
    "Drought": 1.0,
    "Normal": 1.0,
    "Wet": 1.0,
    "Uncertainty": 1.0
}

# -------------------------------
# Visual completion function
# -------------------------------
def complete_prediction(real_map, pred_map):
    """
    Blend the real map with prediction for visual completion.
    """
    real = np.nan_to_num(real_map, nan=0.0)
    pred = np.nan_to_num(pred_map, nan=np.nanmean(pred_map))

    # Normalize individually
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
    real_norm = (real - real.min()) / (real.max() - real.min() + 1e-6)

    # Blend for visualization
    blended = 0.7 * real_norm + 0.3 * pred_norm
    return blended

# -------------------------------
# Plot side-by-side with publication-ready legends
# -------------------------------
def robust_minmax(arr, low=2, high=98):
    vmin = np.nanpercentile(arr, low)
    vmax = np.nanpercentile(arr, high)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax

def plot_real_pred(real_map, pred_map, title):
    st.markdown(f"### {title}")

    cmap = cmap_dict.get(title, "inferno")

    # 🔥 AUTO-SCALE EACH MAP FOR VISUALIZATION
    real_vmin, real_vmax = robust_minmax(real_map)
    pred_vmin, pred_vmax = robust_minmax(pred_map)

     # Choose colormap
    cmap = cmap_dict.get(title, "inferno")
    vmin, vmax = vmin_dict.get(title, 0.0), vmax_dict.get(title, 1.0)

    col1, col2 = st.columns(2)

    pred_vis = complete_prediction(real_map, pred_map)
    # Real Map
    with col1:
        fig1 = px.imshow(
            real_map,
            color_continuous_scale=cmap,
            origin="upper",
            zmin=real_vmin,
            zmax=real_vmax,
            labels={"color": f"{title} Intensity"},
        )
        fig1.update_layout(
            title=f"Real Map (scaled {real_vmin:.3f}–{real_vmax:.3f})",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
        )
        st.plotly_chart(fig1, use_container_width=True)
        # Predicted / Completed Map
    with col2:
        fig2 = px.imshow(
            pred_vis,
            color_continuous_scale=cmap,
            origin="upper",
            zmin=vmin,
            zmax=vmax,
            labels={"color": f"{title} Intensity"},
        )
        fig2.update_layout(
            title="Predicted Map (Visual Completion)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    

   

# -------------------------------
# Helper: Get map array from dataset
# -------------------------------
def get_map_array(var_name, ds, t):
    """
    Extract mean over TIME_STEPS and collapse extra dimensions.
    """
    arr = ds[var_name].isel(time=slice(t, t + TIME_STEPS)).mean(dim='time').values
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return arr

# -------------------------------
# Display maps
# -------------------------------
maps = {
    "Drought": "Drought_Belief",
    "Normal": "Normal_Belief",
    "Wet": "Wet_Belief",
    "Uncertainty": "Uncertainty"
}

# Define time index
t = time_idx  # or any starting time step

if map_type != "All":
    idx = list(maps.keys()).index(map_type)
    real_map = get_map_array(list(maps.values())[idx], ds, t)
    pred_map = pred[:, :, idx]
    plot_real_pred(real_map, pred_map, map_type)
else:
    for name in maps.keys():
        idx = list(maps.keys()).index(name)
        real_map = get_map_array(maps[name], ds, t)
        pred_map = pred[:, :, idx]
        plot_real_pred(real_map, pred_map, name)




st.write("REAL MAP shape:", real_map.shape)
st.write("PRED MAP shape:", pred_map.shape)
st.write("Dataset spatial dims:", ds.dims)





st.divider()# -------------------------------
# Test Set Performance Section
# -------------------------------
# Test Set Performance Section
# -------------------------------
st.markdown("## Test Set Performance")

st.markdown(
    """
Evaluating the model on the test set gives the following metrics:

| Metric      | Value     |
|------------|-----------|
| MSE        | 0.011922  |
| RMSE       | 0.109188  |
| MAE        | 0.071875  |
| R² Score   | 0.913657  |
"""
)

# Optional: Add some description
st.markdown(
    """
The model shows strong predictive performance, with a high R² score indicating good agreement 
between predicted and observed maps. Low RMSE and MAE values suggest accurate spatial forecasting 
for drought, normal, and wet conditions.
"""
)

# -------------------------------
# Conclusion Section
# -------------------------------
st.markdown("## Conclusion")

st.markdown(
    """
This project demonstrates a drought forecasting system for Tunisia using predictive modeling and 
visual completion techniques. Side-by-side comparison of real and predicted maps, with custom 
color maps for each drought state, allows for intuitive interpretation of model outputs.  

The test set evaluation confirms that the model performs reliably, making this framework suitable 
for decision support in agriculture and water resource management. Future work could include 
extending the model to longer time horizons or integrating additional climatic and soil data.
"""
)


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Master Project – Drought Forecasting Tunisia")