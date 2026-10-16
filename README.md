# Clasificador de personajes de Los Simpson

CNN propia (4 bloques Conv + BatchNorm + ReLU + MaxPooling, GAP, Dropout) entrenada
con 10 personajes. App Streamlit que recibe una imagen, la preprocesa
(RGB, 128×128, escala [0, 1]) y devuelve el personaje predicho con su probabilidad.

## Estructura

```
simpsons-classifier/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── model/
    ├── modelo_final.keras   # copiar desde el notebook / Drive (no incluido)
    └── class_names.json
```

## 1. Colocar el modelo

Copia `M1_cnn_lr1e-3.keras` (Drive → `Ev2_models/`) a `model/modelo_final.keras`.

## 2. Ejecutar sin Docker

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. Ejecutar con Docker

```bash
docker compose up --build
```

Abrir http://localhost:8501

## Parámetros opcionales (variables de entorno)

| Variable | Por defecto | Descripción |
|---|---|---|
| `MODEL_PATH` | `model/modelo_final.keras` | Ruta del modelo |
| `CLASSES_PATH` | `model/class_names.json` | Lista de clases |
| `CONF_THRESHOLD` | `0.60` | Bajo este valor se muestra aviso de baja confianza |

## Limitaciones

El modelo solo conoce 10 personajes: ante cualquier otra imagen igualmente elegirá uno.
