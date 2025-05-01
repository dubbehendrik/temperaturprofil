import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Streamlit Layout
st.set_page_config(layout="wide")
st.title("Bestimmung des Wärmeübergangskoeffizienten")

# Hilfsfunktionen
def temperature_model(t, alpha, cp, A, m, T0, T_inf):
    return T_inf - (T_inf - T0) * np.exp(-alpha * A * t / (m * cp))

def calculate_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Abschnitt 1: Dateiupload
uploaded_file = st.file_uploader("Lade eine Excel-Datei hoch", type=["xlsx"])

# Reset bei Datei-Löschen
if uploaded_file is None and "df" in st.session_state:
    st.session_state.clear()
    st.rerun()

# Daten einlesen
if uploaded_file and "df" not in st.session_state:
    df_raw = pd.read_excel(uploaded_file)
    times = df_raw.iloc[:, 0].dropna().values
    temps = df_raw.iloc[:, 1].dropna().values
    min_len = min(len(times), len(temps))
    times = times[:min_len]
    temps = temps[:min_len]
    st.session_state.df = pd.DataFrame({"Zeit_s": times, "Temperatur_C": temps})

    # Parameter auslesen
    params = pd.read_excel(uploaded_file, usecols=[5], skiprows=1, nrows=5, header=None)
    st.session_state.cp = float(params.iloc[0, 0])
    st.session_state.A = float(params.iloc[1, 0])
    st.session_state.m = float(params.iloc[2, 0])
    st.session_state.T0 = float(params.iloc[3, 0])
    st.session_state.T_inf = float(params.iloc[4, 0])

if "df" in st.session_state:
    df = st.session_state.df

    col_plot, col_inputs = st.columns([0.65, 0.35])

    with col_plot:
        st.subheader("Temperaturverlauf")

        time_min = float(df['Zeit_s'].min())
        time_max = float(df['Zeit_s'].max())
        time_range = st.slider("Wähle den betrachteten Zeitbereich:",
                               min_value=time_min,
                               max_value=time_max,
                               value=(time_min, time_max),
                               step=1.0)

        df_cut = df[(df['Zeit_s'] >= time_range[0]) & (df['Zeit_s'] <= time_range[1])].copy()
        df_cut['Zeit_s'] = df_cut['Zeit_s'] - df_cut['Zeit_s'].min()

        fig, ax = plt.subplots()
        ax.plot(df_cut['Zeit_s'], df_cut['Temperatur_C'], 'ro', label="Experiment")
        ax.set_xlabel("Zeit [s]")
        ax.set_ylabel("Temperatur [°C]")
        ax.set_title("Temperaturverlauf")
        ax.legend()
        st.pyplot(fig)

    with col_inputs:
        st.subheader("Parameter")
        
        cp = st.number_input("Wärmekapazität $c_p$ $[J/(kgK)]$", value=st.session_state.cp)
        A = st.number_input("Oberfläche A $[m^2]$", value=st.session_state.A)
        m = st.number_input("Masse m $[kg]$", value=st.session_state.m)
        T0 = st.number_input("Anfangstemperatur $T_0$ [°C]", value=st.session_state.T0)
        T_inf = st.number_input("Umgebungstemperatur $T_\infty$ [°C]", value=st.session_state.T_inf)
            
        calculate_clicked = st.button("Calculate")

        if calculate_clicked:
            # Fit durchführen
            t_data = df_cut['Zeit_s'].values
            T_data = df_cut['Temperatur_C'].values

            try:
                popt, _ = curve_fit(lambda t, alpha: temperature_model(t, alpha, cp, A, m, T0, T_inf),
                                    t_data, T_data, p0=[10.0], bounds=(0, np.inf))
                alpha_fit = popt[0]

                T_fit = temperature_model(t_data, alpha_fit, cp, A, m, T0, T_inf)
                r_squared = calculate_r_squared(T_data, T_fit)
                rmse = calculate_rmse(T_data, T_fit)

                fig2, ax2 = plt.subplots()
                ax2.plot(t_data, T_data, 'ro', label="Experiment")
                ax2.plot(t_data, T_fit, 'b-', label="Simulation")
                ax2.set_xlabel("Zeit [s]")
                ax2.set_ylabel("Temperatur [°C]")
                ax2.set_title("Temperaturverlauf mit Fit")
                ax2.legend()
                ax2.text(0.05, 0.8,
                        f"$\\alpha_{{fit}}$ = {alpha_fit:.2f} $\\frac{{W}}{{m^2K}}$\n$R^2$ = {r_squared:.4f}\nRMSE = {rmse:.2f} \u00b0C",
                        transform=ax2.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                st.pyplot(fig2)

                with st.expander("\u2753 Hilfe zur Interpretation"):
                    st.markdown(r'''
                    **R² (Bestimmtheitsmaß)**:
                    - Gibt an, wie gut das Modell die Daten erklärt.
                    - Werte nahe 1 bedeuten eine sehr gute Anpassung.

                    **RMSE (Root Mean Square Error)**:
                    - Gibt die durchschnittliche Abweichung zwischen Messung und Modell an.
                    - Je kleiner der RMSE, desto besser die Anpassung.

                    **Wichtig:**
                    - Ein hoher R² heißt nicht automatisch niedrigen RMSE.
                    - RMSE hilft, die tatsächliche Abweichung in realen Einheiten (hier °C) zu verstehen.
                    ''')

            except Exception as e:
                st.error(f"Fehler beim Fit: {e}")
