import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Calculation function (unchanged)
def calculate_Lmax_est_F(Tc, m, h, T60, V, rho, hp, E, nu, sigma, g, rho_0, c0):

    f = np.array([20, 25, 31.5, 40, 50, 63, 80, 100,
                  125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                  1250, 1600, 2000])

    a_weight = np.array([-50.5, -44.7, -39.4, -34.6, -30.2, -26.2,
                         -22.5, -19.1, -16.1, -13.4, -10.9, -8.6,
                         -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1,
                         1.2])
    B = 0.23 * f

    omega = 2 * np.pi * f

    # Step 1 - Establish force input
    Fn = 2 * m * np.sqrt(2 * g * h)  # Magnitude of force
    f2rms = (np.abs(Fn)**2 * B) / 2  # Spectral rms composition of force input

    # Step 2 - Determine power injection into floor
    Bp = (E * hp**3) / (12 * (1 - nu**2))
    Zdp = 8 * np.sqrt(Bp * rho * hp)
    Zm = 1j * omega * m
    Win = f2rms * np.real(1 / (Zdp + Zm))

    # Step 3 - Prediction of noise in a floor directly below impact
    n1 = 0.01 + 1 / np.sqrt(f)
    n2 = 2.2 / (f * T60)
    n12 = (rho_0 * c0 * sigma) / (omega * (rho * hp))
    E2 = (n12 / (n1 * n2)) * (Win / omega)
    Lp = 10 * np.log10(E2 / (10**-12)) - 10 * np.log10(V) + 25.4

    # Step 4 - Contact time correction
    fcut_off = 1.5 / Tc
    Lmax_est_F_Spectra = np.zeros_like(Lp)

    for i, frequency in enumerate(f):
        if frequency > fcut_off:
            correction = correction -4
        else:
            correction = 0
        Lmax_est_F_Spectra[i] = Lp[i] + correction + 10 * np.log10(Tc / 0.1)
    
    LAmax_est_F_Spectra = Lmax_est_F_Spectra + a_weight
    LAmax_est_F = round(10 * np.log10(sum(10**(LAmax_est_F_Spectra / 10))), 1)

    crit_freq = round(((c0**2)/(2*np.pi))*np.sqrt(((rho*hp)/Bp)))
    return f, Lmax_est_F_Spectra, LAmax_est_F, fcut_off

# Streamlit App Setup
st.set_page_config(layout="wide", page_title="ProPG GAG Noise Prediction", initial_sidebar_state="expanded")

# Hide default Streamlit style padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("ProPG GAG Simplified Noise Prediction")

with st.sidebar:
    st.sidebar.title("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("Mass (kg)", value=35.0)
        rho = st.number_input("Floor Density (kg/m³)", value=2300.0)
        nu = st.number_input("Poisson Ratio (ν)", value=0.2)
        T60 = st.number_input("Reverberation Time (s)", value=0.6)
    with col2:
        h = st.number_input("Drop Height (m)", value=1.0)
        E = st.number_input("Young's Modulus (GPa)", value=30.0) * 1e9
        sigma = st.number_input("Radiation Efficiency (σ)", value=1.0)
        V = st.number_input("Room Volume (m³)", value=15.0)
        hp = st.number_input("Slab Thickness (mm)", value=250.0) / 1000

    col3,col4 = st.columns(2)
    with col3:
        Tc_start = st.number_input("Tc Start (ms)", value=1.0)
    with col4:
        Tc_end = st.number_input("Tc End (ms)", value=7.0)

    g = 9.81
    c0 = 343
    rho_0 = 1.21


# Loop over Tc range in 1ms steps
Tc_values = np.arange(Tc_start, Tc_end + 1, 1)
Tc_spectra = []

for Tc_ms in Tc_values:
    Tc_sec = Tc_ms / 1000
    f, Lmax_spectrum, LAmax, fcut = calculate_Lmax_est_F(
        Tc_sec, m, h, T60, V, rho, hp, E, nu, sigma, g, rho_0, c0
    )
    Tc_spectra.append(pd.DataFrame({
        "Frequency (Hz)": f,
        "Lmax (dB)": Lmax_spectrum,
        "Tc": f"{int(Tc_ms)} ms"
    }))

# Combine all spectra into one DataFrame
spectra_df = pd.concat(Tc_spectra, ignore_index=True)

# Plot all spectra with Altair
chart = alt.Chart(spectra_df).mark_line().encode(
    x=alt.X("Frequency (Hz):Q", scale=alt.Scale(type="log", base=10, domain=[20, 2000]),
           axis=alt.Axis(values=list(f), labelExpr='datum.value + " Hz"', ticks=True, grid=True)),
    y=alt.Y("Lmax (dB):Q", scale=alt.Scale(domainMin=50)),
    color=alt.Color("Tc:N", scale=alt.Scale(scheme="category20")),
    tooltip=["Frequency (Hz)", "Lmax (dB)", "Tc"]
).properties(
    height=600,
).configure_legend(
    orient="bottom",
    title=None
)

st.altair_chart(chart, use_container_width=True)
