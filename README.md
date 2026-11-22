
# ECG PDF Risk Assessment App

## Files required

Place these files in the same folder:

- `app.py`
- `requirements.txt`
- `ecg_hybrid_model_outputs.zip`

The model ZIP must contain:

- `ecg_risk_model_levels_1_to_4.pkl`
- `ecg_hybrid_model_config.pkl`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Input

Upload a text-based ECG PDF. Image-only scanned PDFs require OCR first.

## Output

- Five-level ECG screening risk
- Confidence
- Probability for each level
- Extracted ECG parameters
- Possible ECG findings
- Quality-control warnings
- CSV download

## Disclaimer

This project is for ECG screening support and academic demonstration. It is not a medical diagnosis system.
