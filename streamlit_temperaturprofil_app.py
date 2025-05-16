import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO
import requests

# Streamlit Layout
st.set_page_config(layout="wide")

col_title, col_logo = st.columns([6, 1])

with col_title:
    st.title("Bestimmung des Wärmeübergangskoeffizienten")

with col_logo:
    st.image("HSE-Logo.jpg", width=1000)


# Hinweistext
with st.expander("ℹ️ Hinweise zur Verwendung"):
    st.markdown("""
    Diese App dient zur Bestimmung des Wärmeübergangskoeffizienten $\\alpha$ anhand gemessener Temperaturverläufe.
    
    Du hast zwei Möglichkeiten:
    - **Eigene Excel-Datei hochladen** anhand von Template (mit Zeit und Temperaturdaten + Parametern).
    - Oder **Beispieldateien verwenden**, um die Funktion zu testen.

    Die App berechnet per Kurvenfit den optimalen Wert für $\\alpha$ und zeigt zusätzlich $R^2$ und RMSE als Qualitätskennzahlen an.
    """)

# Datei-Upload vom Nutzer
uploaded_file = st.file_uploader("Lade eine Excel-Datei hoch", type=["xlsx"])

# Platzhalter für Beispieldatei (initial None)
example_file = None

# Zwei Beispiel-Buttons
col_demo1, col_demo2, col_demo3 = st.columns(3)

with col_demo1:
    if st.button("Beispiel 1 laden"):
        url = "https://raw.githubusercontent.com/dubbehendrik/temperaturprofil/main/Exp_Temperaturprofil_ideal.xlsx"
        response = requests.get(url)
        if response.status_code == 200:
            example_file = BytesIO(response.content)

with col_demo2:
    if st.button("Beispiel 2 laden"):
        url = "https://raw.githubusercontent.com/dubbehendrik/temperaturprofil/main/Exp_Temperaturprofil_real.xlsx"
        response = requests.get(url)
        if response.status_code == 200:
            example_file = BytesIO(response.content)

with col_demo3:
    with open("Exp_Temperaturprofil_ideal.xlsx", "rb") as f:
        st.download_button("Template herunterladen", f, file_name="Exp_Temperaturprofil_ideal.xlsx")

# --- Jetzt das "echte" file_to_use bestimmen ---
file_to_use = uploaded_file if uploaded_file is not None else example_file

# Reset bei Datei-Löschen
if file_to_use is not None:
    col_file, col_remove = st.columns([8, 2])
    with col_file:
        st.success("Datei geladen: Beispieldaten" if example_file else "Datei geladen: Hochgeladen")
    with col_remove:
        if st.button("❌ Entfernen"):
            if "df" in st.session_state:
                del st.session_state["df"]
            uploaded_file = None
            example_file = None
            st.rerun()

# Daten einlesen
if file_to_use is not None and "df" not in st.session_state:
    df_raw = pd.read_excel(file_to_use)
    times = df_raw.iloc[:, 0].dropna().values
    temps = df_raw.iloc[:, 1].dropna().values
    min_len = min(len(times), len(temps))
    times = times[:min_len]
    temps = temps[:min_len]
    st.session_state.df = pd.DataFrame({"Zeit_s": times, "Temperatur_C": temps})

    # Parameter auslesen
    params = pd.read_excel(file_to_use, usecols=[5], skiprows=1, nrows=5, header=None)
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

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(r"Wärmekapazität $c_p\ \left[\frac{J}{\mathrm{kg}\,K}\right]$", unsafe_allow_html=True)
        with col2:
            cp = st.number_input(label="", value=st.session_state.cp)

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(r"Oberfläche $A$ $[m^2]$", unsafe_allow_html=True)
        with col2:
            A = st.number_input(label="", value=st.session_state.A)

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(r"Masse $m$ $[kg]$", unsafe_allow_html=True)
        with col2:
            m = st.number_input(label="", value=st.session_state.m)

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(r"Anfangstemperatur $T_0$ $[^\circ C]$", unsafe_allow_html=True)
        with col2:
            T0 = st.number_input(label="", value=st.session_state.T0)

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(r"Umgebungstemperatur $T_\infty$ $[^\circ C]$", unsafe_allow_html=True)
        with col2:
            T_inf = st.number_input(label="", value=st.session_state.T_inf)

        calculate_clicked = st.button("Calculate")

        if calculate_clicked:
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
                    ''')

            except Exception as e:
                st.error(f"Fehler beim Fit: {e}")
