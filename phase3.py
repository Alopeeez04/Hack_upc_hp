import os
import sqlite3
from groq import Groq

# ─────────────────────────────────────────────
# BASE DE DATOS
# ─────────────────────────────────────────────

class HistorianDB:
    def __init__(self, db_path="historian.db"):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"No se encuentra {self.db_path}")

    def _query(self, sql, params=()):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def get_latest_state(self, scenario_id: str):
        rows = self._query(
            "SELECT * FROM telemetry WHERE scenario_id=? ORDER BY elapsed_hours DESC LIMIT 1",
            (scenario_id,)
        )
        return rows[0] if rows else {}

    def get_scenario_list(self) -> list:
        rows = self._query("SELECT DISTINCT scenario_id FROM telemetry")
        return [r["scenario_id"] for r in rows]

    def get_time_range(self, scenario_id: str, start_hours: float, end_hours: float):
        return self._query(
            "SELECT * FROM telemetry WHERE scenario_id=? AND elapsed_hours BETWEEN ? AND ? ORDER BY elapsed_hours",
            (scenario_id, start_hours, end_hours)
        )


# ─────────────────────────────────────────────
# AGENTE CON GROQ SDK OFICIAL
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Eres **Casiopea**, el Co-Pilot de diagnóstico de la impresora HP Metal Jet S100.
Usas markdown (negrita, listas, emojis de estado) para estructurar tus respuestas.

Tu función es analizar los datos de telemetría en tiempo real y ayudar al operario a tomar decisiones.
Los datos del estado actual de la máquina te serán proporcionados al inicio de cada mensaje de usuario.

Reglas:
- Si el mensaje es un saludo (hola, buenos días, bon dia, hello, hi, etc.), responde ÚNICAMENTE con una presentación breve y amigable: saluda, di que eres Casiopea, el co-piloto de diagnóstico de la HP Metal Jet S100, y pregunta en qué puedes ayudar. NO analices la telemetría en este caso.
- Si preguntan por un componente específico, profundiza en ese componente.
- Si detectas valores críticos (salud < 0.3), alerta con urgencia y propón acción concreta.
- Si preguntan algo fuera de tu ámbito, redirige amablemente.
- Sé conciso: máximo 5-6 líneas salvo que el usuario pida un informe detallado.
- Nunca inventes datos. Solo usa los que te proporcionan en el contexto.
- Responde SIEMPRE en el idioma que se indique en la instrucción al final del mensaje del usuario. Esto es obligatorio y tiene prioridad sobre todo lo demás.

Componentes que monitorizas:
- **Heater**: Sistema de calefacción y control térmico
- **Nozzle**: Boquilla de inyección de aglutinante
- **Recoater/Blade**: Cuchilla de extensión del polvo metálico
"""

GREETINGS = {
    "en", "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "saludos",
    "bon dia", "bona tarda", "bona nit",
}

GREETING_RESPONSES = {
    "en": (
        "👋 Hello! I'm **Casiopea**, your diagnostic co-pilot for the HP Metal Jet S100.\n\n"
        "I monitor the health of your machine's components in real time — Heater, Nozzle, and Blade — "
        "and help you make maintenance decisions.\n\n"
        "How can I help you today? 🛠️"
    ),
    "es": (
        "👋 ¡Hola! Soy **Casiopea**, tu co-piloto de diagnóstico para la HP Metal Jet S100.\n\n"
        "Monitorizo en tiempo real la salud de los componentes de tu máquina — Heater, Nozzle y Blade — "
        "y te ayudo a tomar decisiones de mantenimiento.\n\n"
        "¿En qué puedo ayudarte hoy? 🛠️"
    ),
    "ca": (
        "👋 Hola! Soc **Casiopea**, el teu co-pilot de diagnòstic per a la HP Metal Jet S100.\n\n"
        "Monitoritzo en temps real la salut dels components de la teva màquina — Heater, Nozzle i Blade — "
        "i t'ajudo a prendre decisions de manteniment.\n\n"
        "En què et puc ajudar avui? 🛠️"
    ),
}


def _format_telemetry(data: dict, scenario_id: str) -> str:
    if not data:
        return f"No hay datos de telemetría disponibles para el escenario {scenario_id}."
    lines = [f"📡 Telemetría en tiempo real — Escenario: {scenario_id}"]
    lines.append(f"- Hora transcurrida: {data.get('elapsed_hours', 'N/D'):.1f} h")
    lines.append(f"- Salud del Heater:  {data.get('heater_health', 'N/D')}")
    lines.append(f"- Salud del Nozzle:  {data.get('nozzle_health', 'N/D')}")
    lines.append(f"- Salud del Blade:   {data.get('blade_health', 'N/D')}")
    known = {"elapsed_hours", "heater_health", "nozzle_health", "blade_health", "scenario_id"}
    for k, v in data.items():
        if k not in known:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


class DiagnosticAgent:
    def __init__(self, db: HistorianDB, model_name=None):
        self._db = db
        self._history: list = []
        api_key = os.environ.get("GROQ_API_KEY", "")
        self._client = Groq(api_key=api_key) if api_key else None

    def reset(self):
        self._history = []

    def _detect_lang(self, message: str) -> str:
        """Detect language from the override instruction appended by app.py."""
        msg = message.lower()
        if "responde obligatòriament en català" in msg or "instrucció crítica" in msg:
            return "ca"
        if "responde únicamente en español" in msg or "instrucción crítica" in msg:
            return "es"
        if "respond only in english" in msg or "critical instruction" in msg:
            return "en"
        return "es"  # default

    def _is_greeting(self, message: str) -> bool:
        """Return True if the core message (before language instruction) is just a greeting."""
        # Strip the language instruction block (starts at double newline before INSTRUCCIÓN/CRITICAL)
        core = message.split("\n\n")[0].strip().lower()
        # Remove punctuation
        core = core.strip("¡!¿?.,;:")
        return core in GREETINGS

    def chat(self, user_message: str, scenario_id: str = "CHAOS") -> str:
        if not self._client:
            return "❌ Falta GROQ_API_KEY. Ejecuta: export GROQ_API_KEY='gsk_...'"

        lang = self._detect_lang(user_message)

        # Intercept greetings — no telemetry, no LLM call needed
        if self._is_greeting(user_message):
            reply = GREETING_RESPONSES.get(lang, GREETING_RESPONSES["es"])
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": reply})
            return reply

        telemetry = self._db.get_latest_state(scenario_id)
        enriched = f"{_format_telemetry(telemetry, scenario_id)}\n\n---\nConsulta del operario: {user_message}"

        self._history.append({"role": "user", "content": enriched})

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._history,
                max_tokens=1024,
                temperature=0.4,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"❌ Error Groq: {e}"

        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return reply


# ─────────────────────────────────────────────
# MONITOR DE ALERTAS
# ─────────────────────────────────────────────

class AlertMonitor:
    def __init__(self, db: HistorianDB):
        self._db = db

    def run(self) -> list:
        alerts = []
        for sid in self._db.get_scenario_list():
            st = self._db.get_latest_state(sid)
            if not st:
                continue
            if st.get("blade_health", 1.0) < 0.3:
                alerts.append(f"🚨 [{sid}] Desgaste Crítico en Recoater (blade={st['blade_health']:.2f})")
            if st.get("heater_health", 1.0) < 0.3:
                alerts.append(f"🔥 [{sid}] Riesgo Térmico Crítico (heater={st['heater_health']:.2f})")
            if st.get("nozzle_health", 1.0) < 0.2:
                alerts.append(f"💧 [{sid}] Obstrucción Severa en Nozzle (nozzle={st['nozzle_health']:.2f})")
        return alerts
