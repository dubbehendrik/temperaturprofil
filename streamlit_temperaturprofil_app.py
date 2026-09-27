import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO
import requests

# Streamlit Layout
st.set_page_config(layout="wide")

col_title, col_logo = st.columns([4, 1])

with col_logo:
    st.image("HSE-Logo.jpg", width=1000)

#with col_title:
st.title("Bestimmung des Wärmeübergangskoeffizienten")

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
        st.download_button("📥 Template herunterladen", f, file_name="Exp_Temperaturprofil_ideal.xlsx")

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
if st.session_state.get("file_to_use") is not None and "df" not in st.session_state:
    df_raw = pd.read_excel(st.session_state.file_to_use)
    if df_raw.shape[1] < 6:
        st.error("Die Excel-Datei benötigt Zeit und Temperatur in Spalte A/B sowie fünf Parameter in Spalte F.")
        st.stop()

    # Fehlende Messwerte müssen paarweise entfernt werden, damit Zeiten und
    # Temperaturen aus derselben Excel-Zeile zusammenbleiben.
    measurements = df_raw.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if len(measurements) < 2 or measurements.iloc[:, 0].nunique() < 2:
        st.error("Die Datei benötigt mindestens zwei gültige Messpunkte mit unterschiedlichen Zeiten.")
        st.stop()
    measurements.columns = ["Zeit_s", "Temperatur_C"]

    params = pd.to_numeric(df_raw.iloc[:5, 5], errors="coerce")
    if len(params) != 5 or not np.isfinite(params).all():
        st.error("In Spalte F müssen cp, A, m, T0 und T∞ als Zahlen stehen.")
        st.stop()
    st.session_state.df = measurements.reset_index(drop=True)
    st.session_state.cp, st.session_state.A, st.session_state.m, st.session_state.T0, st.session_state.T_inf = map(float, params)

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
        # T0 gilt zum Beginn des gesamten Versuchs, nicht zum Beginn des Ausschnitts.
        df_cut['Zeit_s'] = df_cut['Zeit_s'] - time_min

        # --- Plot-Placeholder: Der Plot wird hier immer reingerendert ---
        plot_placeholder = st.empty()

    st.markdown("""
    <style>
    .param-block { margin-bottom: 1rem; }
    .param-label { margin-top: 0.2rem; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)
    
    with col_inputs:
        st.subheader("Parameter")
        
        # Wärmekapazität
        cp = st.number_input(label="", value=st.session_state.cp, key="cp_input")
        st.markdown(r"Wärmekapazität $c_p\ \left[\frac{J}{\mathrm{kg}\,K}\right]$", unsafe_allow_html=True)
        
    
        # Oberfläche
        A = st.number_input(label="", value=st.session_state.A, key="A_input")
        st.markdown(r"Oberfläche $A\ [m^2]$", unsafe_allow_html=True)
        
    
        # Masse
        m = st.number_input(label="", value=st.session_state.m, key="m_input")
        st.markdown(r"Masse $m\ [kg]$", unsafe_allow_html=True)
        
    
        # Anfangstemperatur
        T0 = st.number_input(label="", value=st.session_state.T0, key="T0_input")
        st.markdown(r"Anfangstemperatur $T_0\ [^\circ C]$", unsafe_allow_html=True)
        
    
        # Umgebungstemperatur
        T_inf = st.number_input(label="", value=st.session_state.T_inf, key="Tinf_input")
        st.markdown(r"Umgebungstemperatur $T_\infty\ [^\circ C]$", unsafe_allow_html=True)
        
            
        calculate_clicked = st.button("Calculate")

    # --- Plot jetzt erzeugen ---
    fig, ax = plt.subplots()
    ax.plot(df_cut['Zeit_s'], df_cut['Temperatur_C'], 'ro', label="Experiment")

    # --- Falls Calculate gedrückt: Simulation ergänzen ---
    if calculate_clicked:
        try:
            if cp <= 0 or A <= 0 or m <= 0:
                raise ValueError("cp, A und m müssen positiv sein.")
            if len(df_cut) < 3:
                raise ValueError("Für den Fit werden mindestens drei Messpunkte im gewählten Zeitbereich benötigt.")
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



st.markdown("""---""")

st.subheader("🛠️ Feedback & Support")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <a href="https://github.com/dubbehendrik/temperaturprofil/issues/new?template=bug_report.yml" target="_blank">
        <button style="padding: 0.5rem 1rem; background-color: #e74c3c; color: white; border: none; border-radius: 5px; cursor: pointer;">
            🐞 Bug melden
        </button>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <a href="https://github.com/dubbehendrik/temperaturprofil/issues/new?template=feature_request.yml" target="_blank">
        <button style="padding: 0.5rem 1rem; background-color: #2ecc71; color: white; border: none; border-radius: 5px; cursor: pointer;">
            ✨ Feature anfragen
        </button>
    </a>
    """, unsafe_allow_html=True)


st.markdown("""---""")

st.markdown("""
<div style="font-size: 0.5rem; color: gray; text-align: center; line-height: 1.4;">
<b>Disclaimer:</b><br>
Diese Anwendung dient ausschließlich zu Demonstrations- und Lehrzwecken. 
Es wird keine Gewähr für die Richtigkeit, Vollständigkeit oder Aktualität der bereitgestellten Inhalte übernommen.<br>
Die Nutzung erfolgt auf eigene Verantwortung.<br>
Eine kommerzielle Verwendung ist ausdrücklich nicht gestattet.<br>
Für Schäden materieller oder ideeller Art, die durch die Nutzung der App entstehen, wird keine Haftung übernommen.
<br><br>
<a href="mailto:hendrik.dubbe@hs-esslingen.de?subject=Anfrage%20zu%20Temperaturprofil-App" 
   style="color: gray; text-decoration: none;">
Prof. Dr.-Ing. Hendrik Dubbe 
</a>
</div>
""", unsafe_allow_html=True)
