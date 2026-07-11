# Agriculture Advisor AI Co-Pilot (Streamlit)

This repository contains a Streamlit-based AI Co-Pilot for agriculture diagnostics. It combines a CNN-based leaf disease predictor, soil and weather data, and LLM-generated explanations and recommendations. The app supports a human-in-the-loop workflow and produces a downloadable PDF report.

Quickstart

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. (Optional) Create a `.env` file with API keys if you want live LLM or weather:

```text
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OPENWEATHER_API_KEY=...
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes
- If OpenWeather/OpenAI/Gemini keys are not provided the app will use deterministic mock data and a rule-based fallback for LLM output.
- The repository includes a lightweight mock-trained model (`plant_disease_model.keras`) and pixel-based heuristics for robust demo predictions.

Files of interest
- `app.py` - Streamlit UI and workflow
- `model.py` - Image preprocessing, prediction, Grad-CAM utilities
- `llm_helper.py` - LLM wrappers and fallback disease knowledge base
- `weather_helper.py` - OpenWeather integration with hash-based mock fallback
- `report_helper.py` - PDF generation using ReportLab

For hackathon demos, try the sample leaves in the `samples/` folder.

If you want me to: run tests, add Dockerfile, or package this as an executable, tell me and I'll continue.
