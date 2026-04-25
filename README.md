# 🖨️ HP Metal Jet S100 - Digital Twin & AI Co-Pilot

Este proyecto es una implementación avanzada de un **Gemelo Digital (Digital Twin)** para la impresora 3D industrial **HP Metal Jet S100**. El sistema utiliza modelos físicos, Machine Learning y Procesamiento de Lenguaje Natural (LLMs) para simular el desgaste de componentes críticos y asistir a los operarios en el mantenimiento predictivo.

---

## 🏗️ Arquitectura del Sistema

El proyecto está estructurado en cuatro capas evolutivas:

### 1. ⚙️ Phase 1: Logic Engine (`phase1.py`)
El núcleo matemático del gemelo digital. Implementa los modelos de degradación:
* **Recoater Blade:** Simulación de desgaste abrasivo mediante distribuciones de Weibull.
* **Nozzle Plate:** Fatiga térmica y probabilidad de obstrucción (clogging).
* **Heating Elements:** Modelo de ML basado en `GradientBoostingRegressor` que predice anomalías en la resistencia térmica según el estrés ambiental.

### 2. 📊 Phase 2: Simulation & Historian (`phase2.py`)
El motor de ejecución que genera datos sintéticos de telemetría:
* **Escenarios:** Simula condiciones `NORMAL`, `DIRTY_FACTORY` (alta contaminación) y `CHAOS` (fallos aleatorios).
* **Historian:** Almacena cada paso de la simulación en una base de datos **SQLite** (`historian.db`) y exporta a **CSV**.
* **Visualización:** Genera reportes automáticos de salud y comparación de escenarios.

### 3. 🤖 Phase 3: Diagnostic Agent - Casiopea (`phase3.py`)
La capa de inteligencia artificial que actúa como un Co-Pilot:
* **RAG (Retrieval-Augmented Generation):** El agente consulta la base de datos de telemetría para responder con datos reales.
* **Groq Cloud:** Utiliza el modelo `Llama 3.3 70B` para diagnósticos técnicos.
* **Alert Monitor:** Escanea proactivamente la base de datos en busca de componentes por debajo del 30% de salud.

### 4. 🖥️ Application UI (`app.py`)
Interfaz de usuario moderna desarrollada con **Streamlit**:
* **Dashboard:** Visualización interactiva de la degradación real vs. simulada.
* **Chat con IA:** Interfaz para hablar con "Casiopea" sobre el estado de la máquina.
* **Voice-to-Text:** Soporte para comandos de voz.
* **Multilingüe:** Soporte completo para Español, Inglés y Catalán.

---

## Instalación y Configuración

### 1. Requisitos previos
* Python 3.9 o superior.
* Una API Key de [Groq Cloud](https://console.groq.com/).

### 2. Clonar e Instalar
```bash
git clone [https://github.com/tu-usuario/hp-metal-jet-twin.git](https://github.com/tu-usuario/hp-metal-jet-twin.git)
cd hp-metal-jet-twin

pip install streamlit pandas numpy scikit-learn matplotlib plotly groq streamlit-mic-recorder streamlit-plotly-events
