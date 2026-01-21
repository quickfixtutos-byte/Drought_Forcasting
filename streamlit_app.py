import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import dask.array as da

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



# with st.sidebar:
#     st.header("Controls")

#     t = st.slider(
#         "Select starting time index",
#         0,
#         ds.dims["time"] - 12,
#         0
#     )

#     map_type = st.radio(
#         "Map type",
#         ["Drought", "Normal", "Wet", "Uncertainty", "All"]
#     )

# # -------------------------------
# # Prepare 4 maps
# # -------------------------------
# TIME_STEPS = 12
# def get_map_array(var_name):
#     arr = ds[var_name].isel(time=slice(t, t + TIME_STEPS)).mean(dim='time').values
#     # collapse month dimension if present
#     if arr.ndim == 3:
#         arr = arr.mean(axis=0)
#     return arr

# maps = {
#     "Drought": get_map_array("Drought_Belief"),
#     "Normal": get_map_array("Normal_Belief"),
#     "Wet": get_map_array("Wet_Belief"),
#     "Uncertainty": get_map_array("Uncertainty"),
# }

# cmap_dict = {
#     "Drought": "inferno",
#     "Normal": "viridis",
#     "Wet": "Blues",
#     "Uncertainty": "Greys"
# }

# # -------------------------------
# # Plot function using Plotly
# # -------------------------------
# def plotly_map(data, title, cmap):
#     fig = px.imshow(
#         data,
#         color_continuous_scale=cmap,
#         zmin=0,
#         zmax=1,
#         origin='upper',
#     )
#     fig.update_layout(
#         title=title,
#         xaxis=dict(showticklabels=False),
#         yaxis=dict(showticklabels=False),
#         margin=dict(l=0, r=0, t=30, b=0),
#         height=400,
#     )
#     return fig

# # -------------------------------
# # Display maps
# # -------------------------------
# if map_type != "All":
#     st.plotly_chart(plotly_map(maps[map_type], map_type, cmap_dict[map_type]), use_container_width=True)
# else:
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(plotly_map(maps["Drought"], "Drought", cmap_dict["Drought"]), use_container_width=True)
#         st.plotly_chart(plotly_map(maps["Normal"], "Normal", cmap_dict["Normal"]), use_container_width=True)
#     with col2:
#         st.plotly_chart(plotly_map(maps["Wet"], "Wet", cmap_dict["Wet"]), use_container_width=True)
#         st.plotly_chart(plotly_map(maps["Uncertainty"], "Uncertainty", cmap_dict["Uncertainty"]), use_container_width=True)
# # -------------------------------
# # Footer
# # -------------------------------
# st.markdown("---")
# st.caption("Master Project – Drought Forecasting Tunisia")
