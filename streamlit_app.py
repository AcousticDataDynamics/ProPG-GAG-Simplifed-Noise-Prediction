import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# ProPG GAG Calculation Function
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
    Fn = 2 * m * np.sqrt(2 * g * h)
    f2rms = (np.abs(Fn)**2 * B) / 2

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
st.title("ProPG GAG Simplifed Noise Prediction")

st.markdown("""
    <style>
        /* Remove Streamlit's default top padding */
        .block-container {
            padding-top: 2rem !important;
        }

        /* Remove padding in sidebar */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            padding-top: 0rem !important;
        }

        /* Hide the sidebar collapse (toggle) button */
        section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

#Streamlit Sidebar UI
with st.sidebar:
    st.title("Input Parameters")

    st.subheader("Step 1: Impact Parameters")
    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("Mass (kg)", value=35.0)
    with col2:
        h = st.number_input("Drop Height (m)", value=1.0)

    st.subheader("Step 2: Floor Properties")
    col1, col2 = st.columns(2)
    with col1:
        rho = st.number_input("Floor Density (kg/m³)", value=2300.0)
        nu = st.number_input("Poisson Ratio ν", value=0.2)
    with col2:
        E = st.number_input("Young's Modulus (GPa)", value=30.0) * 1e9
        hp = st.number_input("Slab Thickness (mm)", value=250.0) / 1000

    st.subheader("Step 3: Acoustic Properties")
    col1, col2 = st.columns(2)
    with col1:
        T60 = st.number_input("Reverberation Time (s)", value=0.6)
        rho_0 = st.number_input("Air Density (kg/m³)", value=1.21)
        V = st.number_input("Room Volume (m³)", value=15.0)
    with col2:
        c0 = st.number_input("Speed of Sound (m/s)", value=343)
        sigma = st.number_input("Radiation Efficiency", value=1.0)

    st.subheader("Step 4: Contact")
    col3,col4 = st.columns(2)
    with col3:
        Tc_start = st.number_input("Tc Start (ms)", value=2.0)
    with col4:
        Tc_end = st.number_input("Tc End (ms)", value=7.0)

    g = 9.81

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
    x=alt.X("Frequency (Hz):Q",title="", scale=alt.Scale(type="log", base=10, domain=[20, 2000]),
           axis=alt.Axis(values=list(f), labelExpr='datum.value + " Hz"', ticks=True, grid=True)),
    y=alt.Y("Lmax (dB):Q",
            scale=alt.Scale(domainMin=45)),
    color=alt.Color("Tc:N", scale=alt.Scale(scheme="darkmulti")),
    tooltip=["Frequency (Hz)", "Lmax (dB)", "Tc"]
).properties(
    height=500,
).configure_legend(
    orient="top",
    title=None
)

st.altair_chart(chart, use_container_width=True)

# Collect spectral results at each Tc
spectra_rows = []

for Tc_ms in Tc_values:
    Tc_sec = Tc_ms / 1000
    f, Lmax_spectrum, LAmax, fcut = calculate_Lmax_est_F(
        Tc_sec, m, h, T60, V, rho, hp, E, nu, sigma, g, rho_0, c0
    )
    row = {f"{int(freq) if freq.is_integer() else freq} Hz": round(val, 1) for freq, val in zip(f, Lmax_spectrum)}
    row["Tc"] = f"{int(Tc_ms)} ms"
    row["LAmax (dB)"] = LAmax
    row["fcut-off (Hz)"] = round(fcut, 1)
    spectra_rows.append(row)

# Convert to DataFrame
spectra_table = pd.DataFrame(spectra_rows)

# Sort frequency columns numerically
freq_cols = [col for col in spectra_table.columns if "Hz" in col and "cut" not in col and "LAmax" not in col]
freq_cols_sorted = sorted(freq_cols, key=lambda x: float(x.split()[0]))

# Reorder columns: Tc, Frequencies..., LAmax, fcut-off
ordered_cols = ["Tc"] + freq_cols_sorted + ["LAmax (dB)", "fcut-off (Hz)"]
spectra_table = spectra_table[ordered_cols]

# Display final table
st.dataframe(spectra_table, use_container_width=True, hide_index=True)

# Convert the final table to CSV format
csv = spectra_table.to_csv(index=False).encode("utf-8")

# Display download button in the sidebar
with st.sidebar:
    st.markdown("---")
    st.download_button(
        label="⬇ Download Spectra Table as CSV",
        data=csv,
        file_name="ProPG_GAG_Simplifed_Noise_Predict_Results.csv",
        mime="text/csv"
    )
