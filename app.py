import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from streamlit_plotly_events import plotly_events
from streamlit_mic_recorder import speech_to_text
 
from phase3 import HistorianDB, DiagnosticAgent, AlertMonitor
 
# ─────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "page_title": "HP Metal Jet - Casiopea",
        "sidebar_title": "HP Digital Twin",
        "scenario_label": "🌍 Scenario",
        "alerts_header": "### ⚠️ Alerts",
        "no_alerts": "No critical alerts",
        "reset_chat": "🗑️ Reset chat",
        "main_title": "🛡️ Casiopea Digital Twin",
        "inputs_title": "🌡️ Input Drivers (Simulation)",
        "temp_label": "Temperature (°C)",
        "humidity_label": "Humidity (%)",
        "load_label": "Production Load",
        "load_options": ["Low", "Medium", "High"],
        "status_normal": "🟢 NORMAL",
        "status_degraded": "🟡 DEGRADED",
        "status_critical": "🔴 CRITICAL",
        "system_status": "### System Status: {}",
        "graph_title": "📈 Real vs Simulated Degradation",
        "blade_label": "Blade",
        "nozzle_label": "Nozzle",
        "heater_label": "Heater",
        "real_suffix": "(Real)",
        "sim_suffix": "(Sim)",
        "timeline_title": "Component Degradation Timeline",
        "state_healthy": "Healthy",
        "state_degraded": "Degraded",
        "state_critical": "Critical",
        "highest_risk": "⚠️ Highest risk: {}",
        "failure_critical": "🚨 {} failure in ~{}h",
        "failure_warning": "⚠️ {} failure in ~{}h",
        "failure_info": "{} failure in ~{}h",
        "stress_temp": "High temperature stressing heater",
        "stress_humidity": "Humidity increasing nozzle risk",
        "stress_load": "High load accelerating blade wear",
        "chat_title": "🤖 Casiopea",
        "chat_placeholder": "Ask a question...",
        "mic_language": "en-US",
        "lang_selector": "🌐 Language",
        "machine_view_title": "🖥️ Machine View",
        "diagnostics_title": "🧠 Diagnostics",
        "component_status_label": "Component Status",
        "component_label": "Component",
        "cause_label": "Cause",
        "action_label": "Action",
        "normal_conditions": "Normal conditions",
        "cause_temp": "High temperature",
        "cause_humidity": "High humidity",
        "cause_load": "High load",
        "action_blade": "Replace blade",
        "action_nozzle": "Clean nozzle",
        "action_heater": "Check heating system",
        "failure_warn": "⚠️ Failure in ~{}h",
        "caption": "Solid = real | Dashed = simulated",
    },
    "es": {
        "page_title": "HP Metal Jet - Casiopea",
        "sidebar_title": "HP Digital Twin",
        "scenario_label": "🌍 Escenario",
        "alerts_header": "### ⚠️ Alertas",
        "no_alerts": "Sin alertas críticas",
        "reset_chat": "🗑️ Reiniciar chat",
        "main_title": "🛡️ Casiopea Digital Twin",
        "inputs_title": "🌡️ Parámetros de Entrada (Simulación)",
        "temp_label": "Temperatura (°C)",
        "humidity_label": "Humedad (%)",
        "load_label": "Carga de Producción",
        "load_options": ["Baja", "Media", "Alta"],
        "status_normal": "🟢 NORMAL",
        "status_degraded": "🟡 DEGRADADO",
        "status_critical": "🔴 CRÍTICO",
        "system_status": "### Estado del sistema: {}",
        "graph_title": "📈 Degradación Real vs Simulada",
        "blade_label": "Cuchilla",
        "nozzle_label": "Boquilla",
        "heater_label": "Calentador",
        "real_suffix": "(Real)",
        "sim_suffix": "(Sim)",
        "timeline_title": "Línea de Tiempo de Degradación",
        "state_healthy": "Saludable",
        "state_degraded": "Degradado",
        "state_critical": "Crítico",
        "highest_risk": "⚠️ Mayor riesgo: {}",
        "failure_critical": "🚨 Fallo en {} en ~{}h",
        "failure_warning": "⚠️ Fallo en {} en ~{}h",
        "failure_info": "Fallo en {} en ~{}h",
        "stress_temp": "Alta temperatura estresando el calentador",
        "stress_humidity": "Humedad aumentando el riesgo de la boquilla",
        "stress_load": "Carga alta acelerando el desgaste de la cuchilla",
        "chat_title": "🤖 Casiopea",
        "chat_placeholder": "Haz una pregunta...",
        "mic_language": "es-ES",
        "lang_selector": "🌐 Idioma",
        "machine_view_title": "🖥️ Vista de Máquina",
        "diagnostics_title": "🧠 Diagnóstico",
        "component_status_label": "Estado del Componente",
        "component_label": "Componente",
        "cause_label": "Causa",
        "action_label": "Acción",
        "normal_conditions": "Condiciones normales",
        "cause_temp": "Alta temperatura",
        "cause_humidity": "Alta humedad",
        "cause_load": "Carga alta",
        "action_blade": "Reemplazar cuchilla",
        "action_nozzle": "Limpiar boquilla",
        "action_heater": "Revisar sistema de calefacción",
        "failure_warn": "⚠️ Fallo en ~{}h",
        "caption": "Sólido = real | Punteado = simulado",
    },
    "ca": {
        "page_title": "HP Metal Jet - Casiopea",
        "sidebar_title": "HP Digital Twin",
        "scenario_label": "🌍 Escenari",
        "alerts_header": "### ⚠️ Alertes",
        "no_alerts": "Sense alertes crítiques",
        "reset_chat": "🗑️ Reiniciar xat",
        "main_title": "🛡️ Casiopea Digital Twin",
        "inputs_title": "🌡️ Paràmetres d'Entrada (Simulació)",
        "temp_label": "Temperatura (°C)",
        "humidity_label": "Humitat (%)",
        "load_label": "Càrrega de Producció",
        "load_options": ["Baixa", "Mitjana", "Alta"],
        "status_normal": "🟢 NORMAL",
        "status_degraded": "🟡 DEGRADAT",
        "status_critical": "🔴 CRÍTIC",
        "system_status": "### Estat del sistema: {}",
        "graph_title": "📈 Degradació Real vs Simulada",
        "blade_label": "Fulla",
        "nozzle_label": "Broquet",
        "heater_label": "Escalfador",
        "real_suffix": "(Real)",
        "sim_suffix": "(Sim)",
        "timeline_title": "Línia de Temps de Degradació",
        "state_healthy": "Saludable",
        "state_degraded": "Degradat",
        "state_critical": "Crític",
        "highest_risk": "⚠️ Major risc: {}",
        "failure_critical": "🚨 Fallada a {} en ~{}h",
        "failure_warning": "⚠️ Fallada a {} en ~{}h",
        "failure_info": "Fallada a {} en ~{}h",
        "stress_temp": "Alta temperatura estressant l'escalfador",
        "stress_humidity": "Humitat augmentant el risc del broquet",
        "stress_load": "Càrrega alta accelerant el desgast de la fulla",
        "chat_title": "🤖 Casiopea",
        "chat_placeholder": "Fes una pregunta...",
        "mic_language": "ca-ES",
        "lang_selector": "🌐 Idioma",
        "machine_view_title": "🖥️ Vista de Màquina",
        "diagnostics_title": "🧠 Diagnòstic",
        "component_status_label": "Estat del Component",
        "component_label": "Component",
        "cause_label": "Causa",
        "action_label": "Acció",
        "normal_conditions": "Condicions normals",
        "cause_temp": "Alta temperatura",
        "cause_humidity": "Alta humitat",
        "cause_load": "Càrrega alta",
        "action_blade": "Substituir la fulla",
        "action_nozzle": "Netejar el broquet",
        "action_heater": "Revisar el sistema de calefacció",
        "failure_warn": "⚠️ Fallada en ~{}h",
        "caption": "Sòlid = real | Puntejat = simulat",
    },
}
 
LANG_OPTIONS = {
    "English": "en",
    "Español": "es",
    "Català": "ca",
}
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HP Metal Jet - Casiopea",
    layout="wide",
    page_icon="🖨️"
)
 
# SIDEBAR STYLE
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f3e5ab;
}
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label {
    color: #4b3621 !important;
}
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
@st.cache_resource
def init():
    db = HistorianDB()
    agent = DiagnosticAgent(db)
    monitor = AlertMonitor(db)
    return db, agent, monitor
 
db, agent, monitor = init()
 
# ─────────────────────────────────────────────
# LANGUAGE SELECTION (sidebar, top)
# ─────────────────────────────────────────────
with st.sidebar:
    st.sidebar.image("foto.png", width=100)
 
    # Language selector first so T() works for everything below
    selected_lang_label = st.selectbox(
        "🌐 Language / Idioma",
        list(LANG_OPTIONS.keys()),
        index=1  # Default: Español
    )
    lang = LANG_OPTIONS[selected_lang_label]
    T = TRANSLATIONS[lang]  # shortcut
 
    st.title(T["sidebar_title"])
 
    scenario = st.selectbox(T["scenario_label"], db.get_scenario_list())
 
    alerts = monitor.run()
    if alerts:
        st.markdown(T["alerts_header"])
        for a in alerts:
            st.error(a)
    else:
        st.success(T["no_alerts"])
 
    if st.button(T["reset_chat"]):
        st.session_state.messages = []
        st.rerun()
 
# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title(T["main_title"])
st.caption(T["caption"])
 
# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
st.subheader(T["inputs_title"])
 
c1, c2, c3 = st.columns(3)
temp = c1.slider(T["temp_label"], 20, 40, 30)
humidity = c2.slider(T["humidity_label"], 30, 90, 60)
load_options = T["load_options"]
load = c3.selectbox(T["load_label"], load_options)
 
# Normalize load to English for internal logic
load_index = load_options.index(load)
load_en = ["Low", "Medium", "High"][load_index]
 
# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
data = db.get_time_range(scenario, 0, 2000)
df_real = pd.DataFrame(data) if data else pd.DataFrame()
 
if df_real.empty:
    st.error("No data available")
    st.stop()
 
# ─────────────────────────────────────────────
# SIMULACIÓN
# ─────────────────────────────────────────────
def apply_stress(df, temp, humidity, load_en):
    df = df.copy()
    factor = 1.0
 
    if temp > 32: factor += 0.15
    if humidity > 70: factor += 0.15
    if load_en == "High": factor += 0.2
    if load_en == "Low": factor -= 0.1
 
    for col in ["blade_health", "nozzle_health", "heater_health"]:
        df[col] = df[col] ** factor
 
    return df
 
df_sim = apply_stress(df_real, temp, humidity, load_en)
 
# ─────────────────────────────────────────────
# COMPONENT NAMES (translated)
# ─────────────────────────────────────────────
comp_names = {
    "blade": T["blade_label"],
    "nozzle": T["nozzle_label"],
    "heater": T["heater_label"],
}
 
# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
state = db.get_latest_state(scenario)
if not state:
    st.error("No state available")
    st.stop()
 
values = {
    T["blade_label"]: state["blade_health"],
    T["nozzle_label"]: state["nozzle_health"],
    T["heater_label"]: state["heater_health"],
}
 
# Also keep English keys for internal logic
values_en = {
    "Blade": state["blade_health"],
    "Nozzle": state["nozzle_health"],
    "Heater": state["heater_health"],
}
 
worst_en = min(values_en, key=values_en.get)
 
# ─────────────────────────────────────────────
# KPI + STATUS (REAL)
# ─────────────────────────────────────────────
avg = (state['blade_health'] + state['nozzle_health'] + state['heater_health']) / 3
if avg > 0.7:
    status = T["status_normal"]
elif avg > 0.3:
    status = T["status_degraded"]
else:
    status = T["status_critical"]
 
st.markdown(T["system_status"].format(status))
 
col1, col2, col3 = st.columns(3)
col1.metric(T["blade_label"], f"{state['blade_health']:.2f}")
col2.metric(T["nozzle_label"], f"{state['nozzle_health']:.2f}")
col3.metric(T["heater_label"], f"{state['heater_health']:.2f}")
 
st.markdown("---")
 
# ─────────────────────────────────────────────
# INTERACTION STATE
# ─────────────────────────────────────────────
if "selected_component" not in st.session_state:
    st.session_state.selected_component = worst_en
 
# ─────────────────────────────────────────────
# LAYOUT PRO
# ─────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])
 
# ─────────────────────────────────────────────
# MACHINE VIEW (PIXEL PERFECT + CLICK)
# ─────────────────────────────────────────────
with col_left:
    st.subheader(T["machine_view_title"])
 
    img = Image.open("printer.png")
    width, height = img.size
 
    fig = go.Figure()
 
    fig.add_layout_image(
        dict(
            source=img,
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            xref="x",
            yref="y",
            layer="below"
        )
    )
 
    # ─── POSICIONES EN PIXELES (AJUSTADAS) ───
    COMPONENT_POS_PX = {
        "Blade": (int(width * 0.50), int(height * 0.62)),
        "Nozzle": (int(width * 0.47), int(height * 0.40)),
        "Heater": (int(width * 0.72), int(height * 0.55))
    }
 
    for comp_en, (x_px, y_px_from_top) in COMPONENT_POS_PX.items():
        y = height - y_px_from_top
        x = x_px
 
        is_selected = comp_en == st.session_state.selected_component
        is_worst = comp_en == worst_en
 
        comp_translated = comp_names[comp_en.lower()]
 
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(
                size=32 if is_selected else 16,
                color="red" if is_worst else "#00D4FF",
                opacity=1 if is_selected else 0.8,
                line=dict(width=2, color="white")
            ),
            text=[comp_translated],
            textposition="top center",
            customdata=[comp_en],
            hovertemplate=f"<b>{comp_translated}</b><br>Health: {values_en[comp_en]:.2f}<extra></extra>",
            showlegend=False
        ))
 
    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height], scaleanchor="x")
 
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
 
    # Click interactivo
    selected = plotly_events(fig, click_event=True, override_height=520)
 
    if selected:
        idx = selected[0]["pointIndex"]
        comp_en = list(COMPONENT_POS_PX.keys())[idx]
        st.session_state.selected_component = comp_en
 
# ─────────────────────────────────────────────
# DIAGNÓSTICO DINÁMICO
# ─────────────────────────────────────────────
with col_right:
    st.subheader(T["diagnostics_title"])
 
    comp_en = st.session_state.selected_component
    comp_translated = comp_names[comp_en.lower()]
    value = values_en[comp_en]
 
    status_comp = T["status_normal"] if value > 0.7 else T["status_degraded"] if value > 0.3 else T["status_critical"]
 
    st.metric(T["component_status_label"], status_comp)
    st.write(f"**{T['component_label']}:** {comp_translated}")
 
    causes = []
    if temp > 32: causes.append(T["cause_temp"])
    if humidity > 70: causes.append(T["cause_humidity"])
    if load_en == "High": causes.append(T["cause_load"])
 
    st.markdown(f"**{T['cause_label']}**")
    for c in causes or [T["normal_conditions"]]:
        st.write(f"• {c}")
 
    actions = {
        "Blade": T["action_blade"],
        "Nozzle": T["action_nozzle"],
        "Heater": T["action_heater"],
    }
 
    st.markdown(f"**{T['action_label']}**")
    st.success(actions[comp_en])
 
# ─────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────
def predict_failure(df, comp):
    if len(df) < 10:
        return None
 
    last = df.iloc[-1]
    prev = df.iloc[-10]
 
    dt = last["elapsed_hours"] - prev["elapsed_hours"]
    if dt == 0:
        return None
 
    slope = (last[f"{comp}_health"] - prev[f"{comp}_health"]) / dt
    if slope >= 0:
        return None
 
    return (last[f"{comp}_health"] - 0.3) / abs(slope)
 
t = predict_failure(df_sim, comp_en.lower())
if t:
    st.warning(T["failure_warn"].format(f"{t:.0f}"))
 
# ─────────────────────────────────────────────
# 📈 REAL vs SIMULATED GRAPH
# ─────────────────────────────────────────────
st.subheader(T["graph_title"])
 
colors = {
    "blade": "#0096D6",
    "nozzle": "#FF4B4B",
    "heater": "#FFAA00"
}
 
fig2 = go.Figure()
 
for c in ["blade", "nozzle", "heater"]:
    comp_label = comp_names[c]
    fig2.add_trace(go.Scatter(
        x=df_real["elapsed_hours"],
        y=df_real[f"{c}_health"],
        name=f"{comp_label} {T['real_suffix']}",
        line=dict(color=colors[c])
    ))
 
    fig2.add_trace(go.Scatter(
        x=df_sim["elapsed_hours"],
        y=df_sim[f"{c}_health"],
        name=f"{comp_label} {T['sim_suffix']}",
        line=dict(color=colors[c], dash="dot")
    ))
 
fig2.add_hline(y=0.3, line_dash="dash", line_color="red")
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)
 
# ─────────────────────────────────────────────
# PREDICCIÓN GLOBAL + INSIGHT
# ─────────────────────────────────────────────
def predict(df, comp):
    if len(df) < 10:
        return None
 
    last = df.iloc[-1]
    prev = df.iloc[-10]
 
    h_now = last[f"{comp}_health"]
    h_prev = prev[f"{comp}_health"]
 
    dt = last["elapsed_hours"] - prev["elapsed_hours"]
    if dt == 0:
        return None
 
    slope = (h_now - h_prev) / dt
    if slope >= 0:
        return None
 
    t_now = last["elapsed_hours"]
    t_fail = t_now + (h_now - 0.3) / abs(slope)
 
    return t_fail - t_now
 
failures = {}
for c in ["blade", "nozzle", "heater"]:
    tf = predict(df_sim, c)
    if tf:
        failures[c] = tf
 
worst_key = min(["blade_health", "nozzle_health", "heater_health"], key=lambda c: state[c]).replace('_health', '')
st.warning(T["highest_risk"].format(comp_names[worst_key].upper()))
 
if failures:
    worst_fail = min(failures, key=failures.get)
    tf_val = round(failures[worst_fail], 1)
    comp_label = comp_names[worst_fail].upper()
 
    if tf_val < 50:
        st.error(T["failure_critical"].format(comp_label, tf_val))
    elif tf_val < 150:
        st.warning(T["failure_warning"].format(comp_label, tf_val))
    else:
        st.info(T["failure_info"].format(comp_label, tf_val))
 
# ─────────────────────────────────────────────
# TIMELINE (verde / naranja / rojo)
# ─────────────────────────────────────────────
st.subheader(T["timeline_title"])
 
def get_state(v):
    return T["state_healthy"] if v > 0.7 else T["state_degraded"] if v > 0.3 else T["state_critical"]
 
df_t = df_sim.copy()
df_t[T["blade_label"]] = df_t["blade_health"].apply(get_state)
df_t[T["nozzle_label"]] = df_t["nozzle_health"].apply(get_state)
df_t[T["heater_label"]] = df_t["heater_health"].apply(get_state)
 
df_melt = df_t.melt(
    id_vars="elapsed_hours",
    value_vars=[T["blade_label"], T["nozzle_label"], T["heater_label"]],
    var_name="Component",
    value_name="State"
)
 
fig3 = px.scatter(
    df_melt,
    x="elapsed_hours",
    y="Component",
    color="State",
    color_discrete_map={
        T["state_healthy"]: "#00C853",
        T["state_degraded"]: "#FF9100",
        T["state_critical"]: "#D50000"
    }
)
 
fig3.update_layout(template="plotly_dark", height=250)
st.plotly_chart(fig3, use_container_width=True)
 
# ─────────────────────────────────────────────
# 🔎 STRESS WARNINGS
# ─────────────────────────────────────────────
msg = []
if temp > 32:
    msg.append(T["stress_temp"])
if humidity > 70:
    msg.append(T["stress_humidity"])
if load_en == "High":
    msg.append(T["stress_load"])
 
if msg:
    st.warning("⚠️ " + " | ".join(msg))
 
# ─────────────────────────────────────────────
# 🤖 CHAT + VOZ
# ─────────────────────────────────────────────
st.markdown("---")
 
col_title, col_mic = st.columns([5, 1])
with col_title:
    st.subheader(T["chat_title"])
with col_mic:
    text_from_voice = speech_to_text(
        language=T["mic_language"],
        start_prompt="🎙️",
        stop_prompt="⏹️",
        use_container_width=True,
        just_once=True
    )
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 
prompt = st.chat_input(T["chat_placeholder"])
user_input = prompt if prompt else text_from_voice
 
if user_input:
    # Strong language override — must come AFTER the question so the model sees it last
    lang_instructions = {
        "en": (
            "CRITICAL INSTRUCTION — OVERRIDE ALL OTHER RULES: "
            "You MUST respond ONLY in English. "
            "Do NOT use Spanish or Catalan under any circumstances. "
            "Even if your system prompt says otherwise, English is mandatory."
        ),
        "es": (
            "INSTRUCCIÓN CRÍTICA — ANULA CUALQUIER OTRA REGLA: "
            "DEBES responder ÚNICAMENTE en español. "
            "No uses inglés ni catalán bajo ninguna circunstancia."
        ),
        "ca": (
            "INSTRUCCIÓ CRÍTICA — ANUL·LA QUALSEVOL ALTRA REGLA: "
            "Has de respondre OBLIGATÒRIAMENT en català. "
            "No facis servir castellà ni anglès sota cap circumstància. "
            "Encara que el teu system prompt digui una altra cosa, el català és obligatori. "
            "Si la teva resposta no és en català, és incorrecta."
        ),
    }
    enriched_input = f"{user_input}\n\n{lang_instructions[lang]}"
 
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = agent.chat(enriched_input, scenario)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
