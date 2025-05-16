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

Die zugrunde liegende Gleichung lautet:
""")

    st.latex(r"T(t) = T_\infty - (T_\infty - T_0) \cdot e^{- \frac{ \alpha A }{ m c_p } t}")

    st.markdown("""
Du hast zwei Möglichkeiten:
- **Eigene Excel-Datei hochladen** anhand des Templates (mit Zeit- und Temperaturdaten + Parametern).
- Oder **Beispieldaten verwenden**, um die Funktion zu testen.

Die App berechnet per Kurvenfit den optimalen Wert für $\\alpha$ und zeigt zusätzlich $R^2$ und RMSE als Qualitätskennzahlen an.

Wichtig: Wenn eine Excel-Datei geladen wurde muss diese entfernt werden (X) um wieder auf Beispieldaten zu wechseln. 
""")

# --- Datei-Upload vom Nutzer ---
uploaded_file = st.file_uploader("Lade eine Excel-Datei hoch", type=["xlsx"])

# --- User lädt eigene Datei hoch → SessionState setzen & rerun ---
if uploaded_file is not None and uploaded_file != st.session_state.get("uploaded_file"):
    st.session_state.clear()
    st.session_state.uploaded_file = uploaded_file
    st.session_state.file_to_use = uploaded_file
    st.session_state.source_label = uploaded_file.name
    st.rerun()

# --- User klickt im Upload X → SessionState löschen ---
if uploaded_file is None and "file_to_use" in st.session_state and st.session_state.get("uploaded_file") is not None:
    st.session_state.clear()
    st.rerun()

# --- Beispiel-Buttons ---
col_demo1, col_demo2, col_demo3 = st.columns(3)

with col_demo1:
    if st.button("Beispiel 1"):
        st.session_state.clear()
        url = "https://raw.githubusercontent.com/dubbehendrik/temperaturprofil/main/Exp_Temperaturprofil_ideal.xlsx"
        response = requests.get(url)
        if response.status_code == 200:
            st.session_state.file_to_use = BytesIO(response.content)
            st.session_state.source_label = "Beispiel 1 geladen"
            st.session_state.uploaded_file = None
            st.rerun()

with col_demo2:
    if st.button("Beispiel 2"):
        st.session_state.clear()
        url = "https://raw.githubusercontent.com/dubbehendrik/temperaturprofil/main/Exp_Temperaturprofil_real.xlsx"
        response = requests.get(url)
        if response.status_code == 200:
            st.session_state.file_to_use = BytesIO(response.content)
            st.session_state.source_label = "Beispiel 2 geladen"
            st.session_state.uploaded_file = None
            st.rerun()

with col_demo3:
    with open("Exp_Temperaturprofil_ideal.xlsx", "rb") as f:
        st.download_button("Template herunterladen", f, file_name="Exp_Temperaturprofil_ideal.xlsx")

# --- Anzeige welcher Datei geladen ist + Entfernen-Button ---
# --- Anzeige welcher Datei geladen ist ---
if "file_to_use" in st.session_state:
    col_file, col_remove = st.columns([8, 2])
    with col_file:
        st.success(f"{st.session_state.source_label}")
    with col_remove:
        # ❌ Entfernen-Button nur bei Beispieldaten anzeigen
        if st.session_state.get("uploaded_file") is None:
            if st.button("❌ Entfernen"):
                st.session_state.clear()
                st.rerun()


# --- Temperaturmodell Funktion ---
def temperature_model(t, alpha, cp, A, m, T0, T_inf):
    return T_inf - (T_inf - T0) * np.exp(-alpha * A * t / (m * cp))

# --- R² Funktion ---
def calculate_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# --- RMSE Funktion ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- Verarbeitung der Datei wenn vorhanden ---
if "file_to_use" in st.session_state and "df" not in st.session_state:
    file_like = st.session_state.get("file_to_use")
    if file_like is not None:
        df_raw = pd.read_excel(file_like)


if "file_to_use" in st.session_state and st.session_state["file_to_use"] is not None:
    df_raw = pd.read_excel(st.session_state["file_to_use"])
    
    times = df_raw.iloc[:, 0].dropna().values
    temps = df_raw.iloc[:, 1].dropna().values
    min_len = min(len(times), len(temps))
    times = times[:min_len]
    temps = temps[:min_len]
    st.session_state.df = pd.DataFrame({"Zeit_s": times, "Temperatur_C": temps})

    # Parameter auslesen
    params = pd.read_excel(st.session_state.file_to_use, usecols=[5], skiprows=1, nrows=5, header=None)
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

        # --- Plot-Placeholder: Der Plot wird hier immer reingerendert ---
        plot_placeholder = st.empty()

    with col_inputs:
        st.subheader("Parameter")

        st.markdown("&nbsp;", unsafe_allow_html=True)  # fügt Abstand ein
        st.markdown("&nbsp;", unsafe_allow_html=True)  # fügt Abstand ein
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(r"""
            Wärmekapazität<br>
            $c_p\ \left[\frac{J}{\mathrm{kg}\,K}\right]$
            """, unsafe_allow_html=True)
        with col2:
            cp = st.number_input(label="", value=st.session_state.cp)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(r"""
            Oberfläche<br>
            $A\ [m^2]$
            """, unsafe_allow_html=True)
        with col2:
            A = st.number_input(label="", value=st.session_state.A)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(r"""
            Masse<br>
            $m\ [kg]$
            """, unsafe_allow_html=True)
        with col2:
            m = st.number_input(label="", value=st.session_state.m)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(r"""
            Anfangstemperatur<br>
            $T_0\ [^\circ C]$
            """, unsafe_allow_html=True)
        with col2:
            T0 = st.number_input(label="", value=st.session_state.T0)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(r"""
            Umgebungstemperatur<br>
            $T_\infty\ [^\circ C]$
            """, unsafe_allow_html=True)
        with col2:
            T_inf = st.number_input(label="", value=st.session_state.T_inf)

        calculate_clicked = st.button("Calculate")

    # --- Plot jetzt erzeugen ---
    fig, ax = plt.subplots()
    ax.plot(df_cut['Zeit_s'], df_cut['Temperatur_C'], 'ro', label="Experiment")

    # --- Falls Calculate gedrückt: Simulation ergänzen ---
    if calculate_clicked:
        try:
            popt, _ = curve_fit(lambda t, alpha: temperature_model(t, alpha, cp, A, m, T0, T_inf),
                                df_cut['Zeit_s'].values, df_cut['Temperatur_C'].values,
                                p0=[10.0], bounds=(0, np.inf))
            alpha_fit = popt[0]

            T_fit = temperature_model(df_cut['Zeit_s'].values, alpha_fit, cp, A, m, T0, T_inf)
            r_squared = calculate_r_squared(df_cut['Temperatur_C'].values, T_fit)
            rmse = calculate_rmse(df_cut['Temperatur_C'].values, T_fit)

            # --- Fit-Linie hinzufügen ---
            ax.plot(df_cut['Zeit_s'], T_fit, 'b-', label="Simulation")

            # --- Fit-Werte als Textbox ---
            ax.text(0.05, 0.95,
                    f"$\\alpha_{{fit}}$ = {alpha_fit:.2f} $\\frac{{W}}{{m^2K}}$\n$R^2$ = {r_squared:.4f}\nRMSE = {rmse:.2f} °C",
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

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

    # --- Achsen & Legende ---
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Temperatur [°C]")
    ax.set_title("Temperaturverlauf")
    ax.legend()

    # --- Plot anzeigen ---
    plot_placeholder.pyplot(fig)

