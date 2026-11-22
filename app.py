# ============================================================
# ECG RISK ASSESSMENT DASHBOARD
# Streamlit Application
# ============================================================

import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

from pypdf import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ECG Risk Assessment",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

RISK_COLORS = {
    1: "#2E7D32",
    2: "#66BB6A",
    3: "#F9A825",
    4: "#EF6C00",
    5: "#C62828",
}

PARAMETER_NORMAL_RANGES = {
    "Heart Rate": (60, 100),
    "P Duration": (60, 120),
    "PR Interval": (120, 200),
    "QRS Duration": (70, 120),
    "QTc Bazett": (350, 450),
}


# ============================================================
# BASIC STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 700;
    }

    h2, h3 {
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .stAlert {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert a value to float safely."""

    try:

        if value is None:
            return np.nan

        if isinstance(value, str):
            value = value.replace(",", "").strip()

        return float(value)

    except (TypeError, ValueError):

        return np.nan


def clean_text(value):
    """Clean text safely."""

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def is_valid_number(value):
    """Return True when value is a finite number."""

    try:

        return np.isfinite(
            float(value)
        )

    except (TypeError, ValueError):

        return False


def get_risk_name(
    risk_names,
    level,
):
    """Safely obtain risk level name."""

    if isinstance(
        risk_names,
        dict,
    ):

        return risk_names.get(
            level,
            risk_names.get(
                str(level),
                f"Level {level}",
            ),
        )

    if isinstance(
        risk_names,
        (list, tuple),
    ):

        if 1 <= level <= len(
            risk_names
        ):

            return risk_names[
                level - 1
            ]

    return f"Level {level}"


def get_risk_color(level):

    return RISK_COLORS.get(
        level,
        "#616161",
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(
    uploaded_file,
):
    """
    Extract text from an uploaded ECG report PDF.
    """

    try:

        reader = PdfReader(
            uploaded_file
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as exc:

        raise RuntimeError(
            f"Unable to read the ECG report: {exc}"
        )


# ============================================================
# NUMBER EXTRACTION HELPERS
# ============================================================

def find_number(
    text,
    patterns,
):
    """
    Search multiple regex patterns and
    return the first number found.
    """

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except (
                ValueError,
                IndexError,
            ):

                continue

    return np.nan


def find_number_pair(
    text,
    patterns,
):
    """
    Extract two numeric values.
    """

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            try:

                return (
                    float(
                        match.group(1)
                    ),
                    float(
                        match.group(2)
                    ),
                )

            except (
                ValueError,
                IndexError,
            ):

                continue

    return np.nan, np.nan


def find_number_triple(
    text,
    patterns,
):
    """
    Extract three numeric values.
    """

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            try:

                return (
                    float(
                        match.group(1)
                    ),
                    float(
                        match.group(2)
                    ),
                    float(
                        match.group(3)
                    ),
                )

            except (
                ValueError,
                IndexError,
            ):

                continue

    return (
        np.nan,
        np.nan,
        np.nan,
    )


# ============================================================
# ECG PDF PARSER
# ============================================================

def parse_ecg_pdf_text(text):
    """
    Extract ECG report values.

    Supports ECG formats such as:

    P/QRS/T : 67/81/19°

    P: 67 QRS: 81 T: 19

    Axis: P 67 QRS 81 T 19
    """

    text = clean_text(text)

    record = {}

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    # --------------------------------------------------------
    # Heart Rate
    # --------------------------------------------------------

    record["heart_rate_bpm"] = find_number(
        text,
        [
            r"(?:heart\s*rate|ventricular\s*rate|rate)"
            r"\s*[:\-]?\s*(\d+(?:\.\d+)?)",

            r"\bHR\b\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        ],
    )

    # --------------------------------------------------------
    # P Duration
    # --------------------------------------------------------

    record["p_duration_ms"] = find_number(
        text,
        [
            r"P\s*(?:duration)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*ms",

            r"P\s*dur\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)",
        ],
    )

    # --------------------------------------------------------
    # PR Interval
    # --------------------------------------------------------

    record["pr_interval_ms"] = find_number(
        text,
        [
            r"PR\s*(?:interval)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*ms",

            r"PR\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)",
        ],
    )

    # --------------------------------------------------------
    # QRS Duration
    # --------------------------------------------------------

    record["qrs_duration_ms"] = find_number(
        text,
        [
            r"QRS\s*(?:duration)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*ms",

            r"QRS\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)",
        ],
    )

    # ========================================================
    # QT / QTc
    # ========================================================

    qt, qtc = find_number_pair(
        text,
        [
            r"QT\s*/\s*QTc\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*/\s*"
            r"(\d+(?:\.\d+)?)",

            r"QT\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*ms.*?"
            r"QTc\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)",
        ],
    )

    record["qt_interval_ms"] = qt

    record["qtc_bazett_ms"] = qtc

    # --------------------------------------------------------
    # Individual QTc fallback
    # --------------------------------------------------------

    if not is_valid_number(
        record["qtc_bazett_ms"]
    ):

        record["qtc_bazett_ms"] = find_number(
            text,
            [
                r"QTc(?:B|Bazett)?\s*[:\-]?\s*"
                r"(\d+(?:\.\d+)?)",

                r"corrected\s*QT\s*[:\-]?\s*"
                r"(\d+(?:\.\d+)?)",
            ],
        )

    # --------------------------------------------------------
    # Individual QT fallback
    # --------------------------------------------------------

    if not is_valid_number(
        record["qt_interval_ms"]
    ):

        record["qt_interval_ms"] = find_number(
            text,
            [
                r"\bQT\b\s*[:\-]?\s*"
                r"(\d+(?:\.\d+)?)\s*ms",
            ],
        )

    # ========================================================
    # ELECTRICAL AXIS
    # ========================================================

    p_axis = np.nan
    qrs_axis = np.nan
    t_axis = np.nan

    # --------------------------------------------------------
    # PRIMARY FORMAT
    #
    # P/QRS/T : 67/81/19°
    # --------------------------------------------------------

    axis_match = re.search(
        r"P\s*/\s*QRS\s*/\s*T"
        r"\s*[:\-]?\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*/\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*/\s*"
        r"(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if axis_match:

        try:

            p_axis = float(
                axis_match.group(1)
            )

            qrs_axis = float(
                axis_match.group(2)
            )

            t_axis = float(
                axis_match.group(3)
            )

        except (
            ValueError,
            IndexError,
        ):

            p_axis = np.nan
            qrs_axis = np.nan
            t_axis = np.nan

    # --------------------------------------------------------
    # FORMAT:
    #
    # P / QRS / T = 67 / 81 / 19
    # --------------------------------------------------------

    if not all(
        is_valid_number(value)
        for value in [
            p_axis,
            qrs_axis,
            t_axis,
        ]
    ):

        axis_match = re.search(
            r"P\s*/\s*QRS\s*/\s*T"
            r"\s*=\s*"
            r"(-?\d+(?:\.\d+)?)"
            r"\s*/\s*"
            r"(-?\d+(?:\.\d+)?)"
            r"\s*/\s*"
            r"(-?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )

        if axis_match:

            try:

                p_axis = float(
                    axis_match.group(1)
                )

                qrs_axis = float(
                    axis_match.group(2)
                )

                t_axis = float(
                    axis_match.group(3)
                )

            except (
                ValueError,
                IndexError,
            ):

                pass

    # --------------------------------------------------------
    # FORMAT:
    #
    # P 67 QRS 81 T 19
    # --------------------------------------------------------

    if not all(
        is_valid_number(value)
        for value in [
            p_axis,
            qrs_axis,
            t_axis,
        ]
    ):

        (
            p_axis_alt,
            qrs_axis_alt,
            t_axis_alt,
        ) = find_number_triple(
            text,
            [
                r"P\s*[:\-]?\s*"
                r"(-?\d+(?:\.\d+)?)"
                r"\s+"
                r"QRS\s*[:\-]?\s*"
                r"(-?\d+(?:\.\d+)?)"
                r"\s+"
                r"T\s*[:\-]?\s*"
                r"(-?\d+(?:\.\d+)?)",
            ],
        )

        if is_valid_number(
            p_axis_alt
        ):

            p_axis = p_axis_alt

        if is_valid_number(
            qrs_axis_alt
        ):

            qrs_axis = qrs_axis_alt

        if is_valid_number(
            t_axis_alt
        ):

            t_axis = t_axis_alt

    # --------------------------------------------------------
    # Individual axis fallback
    # --------------------------------------------------------

    if not is_valid_number(
        p_axis
    ):

        p_axis = find_number(
            text,
            [
                r"P\s*axis\s*[:\-]?\s*"
                r"(-?\d+(?:\.\d+)?)",
            ],
        )

    if not is_valid_number(
        qrs_axis
    ):

        qrs_axis = find_number(
            text,
            [
                r"QRS\s*axis\s*[:\-]?\s*"
                r"(-?\d+(?:\.\d+)?)",
            ],
        )

    if not is_valid_number(
        t_axis
    ):

        t_axis = find_number(
            text,
            [
                r"T\s*axis\s*[:\-]?\s*"
                r"(-?\d+(?:\.\d+)?)",
            ],
        )

    record["p_axis_deg"] = p_axis

    record["qrs_axis_deg"] = qrs_axis

    record["t_axis_deg"] = t_axis

    # ========================================================
    # RV5 / SV1
    # ========================================================

    record["rv5_mv"] = find_number(
        text,
        [
            r"RV5\s*[:\-]?\s*"
            r"(-?\d+(?:\.\d+)?)",

            r"RV5\s*=\s*"
            r"(-?\d+(?:\.\d+)?)",
        ],
    )

    record["sv1_mv"] = find_number(
        text,
        [
            r"SV1\s*[:\-]?\s*"
            r"(-?\d+(?:\.\d+)?)",

            r"SV1\s*=\s*"
            r"(-?\d+(?:\.\d+)?)",
        ],
    )

    # ========================================================
    # RHYTHM
    # ========================================================

    rhythm_match = re.search(
        r"(?:rhythm|ventricular\s*rhythm|"
        r"atrial\s*rhythm)"
        r"\s*[:\-]\s*"
        r"([^\n,;]+)",
        text,
        flags=re.IGNORECASE,
    )

    if rhythm_match:

        record["rhythm"] = clean_text(
            rhythm_match.group(1)
        )

    else:

        rhythm_patterns = [
            "sinus rhythm",
            "sinus tachycardia",
            "sinus bradycardia",
            "sinus arrhythmia",
            "atrial fibrillation",
            "atrial flutter",
            "ventricular tachycardia",
            "ventricular fibrillation",
        ]

        detected_rhythm = "Unknown"

        lower_text = text.lower()

        for rhythm in rhythm_patterns:

            if rhythm in lower_text:

                detected_rhythm = (
                    rhythm.title()
                )

                break

        record["rhythm"] = (
            detected_rhythm
        )

    # ========================================================
    # PRIMARY LABEL
    # ========================================================

    primary_label = ""

    label_patterns = [
        r"(?:diagnosis|interpretation|"
        r"impression)"
        r"\s*[:\-]\s*([^\n]+)",

        r"(?:primary\s*diagnosis|"
        r"primary\s*label)"
        r"\s*[:\-]\s*([^\n]+)",
    ]

    for pattern in label_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            primary_label = clean_text(
                match.group(1)
            )

            break

    record["primary_label"] = (
        primary_label
        if primary_label
        else "Unknown"
    )

    # ========================================================
    # MACHINE FINDINGS
    # ========================================================

    known_findings = [
        "sinus tachycardia",
        "sinus bradycardia",
        "sinus arrhythmia",
        "sinus rhythm",
        "ptfv1",
        "t wave abnormality",
        "t-wave abnormality",
        "st-t abnormality",
        "lvh",
        "left ventricular hypertrophy",
        "high voltage",
        "poor r wave progression",
        "premature ventricular contraction",
        "premature atrial contraction",
        "pvc",
        "pac",
        "atrial fibrillation",
        "atrial flutter",
        "first degree av block",
        "first-degree av block",
        "rbbb",
        "right bundle branch block",
        "lbbb",
        "left bundle branch block",
        "bbb",
        "prolonged qt",
        "myocardial infarction",
        "stemi",
        "ventricular tachycardia",
        "ventricular fibrillation",
        "complete heart block",
        "third degree block",
        "torsades",
        "asystole",
        "paced rhythm",
        "lead off",
        "electrode reversal",
        "normal ecg",
    ]

    detected_findings = []

    lower_text = text.lower()

    for finding in known_findings:

        if finding in lower_text:

            detected_findings.append(
                finding.title()
            )

    record["machine_findings"] = (
        detected_findings
    )

    record["machine_findings_text"] = (
        "; ".join(
            detected_findings
        )
        if detected_findings
        else ""
    )

    # ========================================================
    # HUMAN-FRIENDLY ALIASES
    # ========================================================

    record["Heart Rate"] = (
        record["heart_rate_bpm"]
    )

    record["P Duration"] = (
        record["p_duration_ms"]
    )

    record["PR Interval"] = (
        record["pr_interval_ms"]
    )

    record["QRS Duration"] = (
        record["qrs_duration_ms"]
    )

    record["QT"] = (
        record["qt_interval_ms"]
    )

    record["QTc Bazett"] = (
        record["qtc_bazett_ms"]
    )

    record["P Axis"] = (
        record["p_axis_deg"]
    )

    record["QRS Axis"] = (
        record["qrs_axis_deg"]
    )

    record["T Axis"] = (
        record["t_axis_deg"]
    )

    record["RV5"] = (
        record["rv5_mv"]
    )

    record["SV1"] = (
        record["sv1_mv"]
    )

    record["Rhythm"] = (
        record["rhythm"]
    )

    record["Primary Label"] = (
        record["primary_label"]
    )

    record["Machine Findings"] = (
        record["machine_findings"]
    )

    record["Machine Findings Text"] = (
        record["machine_findings_text"]
    )

    return record


# ============================================================
# QUALITY CHECK
# ============================================================

def detect_quality_problem(record):
    """
    Detect obvious report/data quality problems.
    """

    problems = []

    findings_text = str(
        record.get(
            "Machine Findings Text",
            "",
        )
    ).lower()

    if "lead off" in findings_text:

        problems.append(
            "Lead-off condition detected in the ECG report."
        )

    if "electrode reversal" in findings_text:

        problems.append(
            "Possible electrode reversal detected."
        )

    numeric_features = [
        "Heart Rate",
        "P Duration",
        "PR Interval",
        "QRS Duration",
        "QT",
        "QTc Bazett",
        "P Axis",
        "QRS Axis",
        "T Axis",
        "RV5",
        "SV1",
    ]

    missing_count = sum(
        not is_valid_number(
            record.get(feature)
        )
        for feature in numeric_features
    )

    threshold = max(
        6,
        int(
            len(numeric_features)
            * 0.70
        ),
    )

    if missing_count >= threshold:

        problems.append(
            "A large number of ECG parameters could not be "
            "extracted automatically. Manual review is recommended."
        )

    return problems


# ============================================================
# LEVEL 5 CRITICAL RULE
# ============================================================

def detect_level5_critical_rule(
    record,
):
    """
    Detect critical ECG findings that force Level 5.
    """

    findings_text = str(
        record.get(
            "Machine Findings Text",
            "",
        )
    ).lower()

    critical_terms = [
        "ventricular fibrillation",
        "ventricular tachycardia",
        "torsades",
        "asystole",
        "stemi",
        "complete heart block",
        "third degree block",
        "myocardial infarction",
    ]

    for term in critical_terms:

        if term in findings_text:

            return True, term

    return False, ""


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_bundle():
    """
    Load ECG model and configuration.
    """

    project_root = (
        Path(__file__).resolve().parent
    )

    model_zip = (
        project_root
        / "ecg_hybrid_model_outputs.zip"
    )

    extraction_dir = (
        project_root
        / ".ecg_model"
    )

    model_file = (
        extraction_dir
        / "ecg_risk_model_levels_1_to_4.pkl"
    )

    config_file = (
        extraction_dir
        / "ecg_hybrid_model_config.pkl"
    )

    # --------------------------------------------------------
    # Extract ZIP
    # --------------------------------------------------------

    if not extraction_dir.exists():

        if not model_zip.exists():

            raise FileNotFoundError(
                "Model ZIP file not found: "
                f"{model_zip}"
            )

        extraction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            model_zip,
            "r",
        ) as zip_ref:

            zip_ref.extractall(
                extraction_dir
            )

    # --------------------------------------------------------
    # Search recursively
    # --------------------------------------------------------

    if not model_file.exists():

        matches = list(
            extraction_dir.rglob(
                "ecg_risk_model_levels_1_to_4.pkl"
            )
        )

        if matches:

            model_file = matches[0]

    if not config_file.exists():

        matches = list(
            extraction_dir.rglob(
                "ecg_hybrid_model_config.pkl"
            )
        )

        if matches:

            config_file = matches[0]

    if not model_file.exists():

        raise FileNotFoundError(
            "ECG risk model file was not found."
        )

    if not config_file.exists():

        raise FileNotFoundError(
            "ECG model configuration file was not found."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(
        model_file
    )

    config = joblib.load(
        config_file
    )

    return model, config


# ============================================================
# PROBABILITY FUNCTIONS
# ============================================================

def normalize_to_five(
    probabilities,
):
    """
    Convert Levels 1-4 probabilities
    into a five-level distribution.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if probabilities.size < 4:

        probabilities = np.pad(
            probabilities,
            (
                0,
                4 - probabilities.size,
            ),
        )

    probabilities = (
        probabilities[:4]
    )

    total = (
        probabilities.sum()
    )

    if total <= 0:

        probabilities = np.array(
            [
                0.25,
                0.25,
                0.25,
                0.25,
            ]
        )

    else:

        probabilities = (
            probabilities / total
        )

    probabilities = (
        probabilities * 0.99
    )

    probabilities = np.append(
        probabilities,
        0.01,
    )

    return probabilities


def critical_distribution():

    return np.array(
        [
            0.01,
            0.01,
            0.01,
            0.07,
            0.90,
        ]
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_record(
    record,
    model,
    config,
):
    """
    Generate ECG risk prediction using
    the exact model feature structure.
    """

    numeric_features = config.get(
        "numeric_features",
        [],
    )

    categorical_features = config.get(
        "categorical_features",
        [],
    )

    all_features = config.get(
        "all_features",
        [],
    )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if not all_features:

        all_features = (
            list(numeric_features)
            + list(categorical_features)
        )

    if not all_features:

        if hasattr(
            model,
            "feature_names_in_",
        ):

            all_features = list(
                model.feature_names_in_
            )

    if not all_features:

        raise ValueError(
            "The model configuration does not contain "
            "any input features."
        )

    # ========================================================
    # BUILD INPUT ROW
    # ========================================================

    row = {}

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for feature in numeric_features:

        value = record.get(
            feature,
            np.nan,
        )

        if value is None:

            value = np.nan

        else:

            try:

                value = float(value)

            except (
                TypeError,
                ValueError,
            ):

                value = np.nan

        row[feature] = value

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for feature in categorical_features:

        value = record.get(
            feature,
            "Unknown",
        )

        if value is None:

            value = "Unknown"

        value = str(
            value
        ).strip()

        if not value:

            value = "Unknown"

        row[feature] = value

    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    for feature in all_features:

        if feature not in row:

            row[feature] = record.get(
                feature,
                np.nan,
            )

    X = pd.DataFrame(
        [
            {
                feature: row.get(
                    feature,
                    np.nan,
                )
                for feature in all_features
            }
        ],
        columns=all_features,
    )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    if hasattr(
        model,
        "predict_proba",
    ):

        raw_probability = (
            model.predict_proba(X)[0]
        )

    else:

        prediction = (
            model.predict(X)[0]
        )

        raw_probability = np.zeros(
            4,
            dtype=float,
        )

        try:

            prediction_int = int(
                prediction
            )

            index = (
                prediction_int - 1
            )

            if 0 <= index < 4:

                raw_probability[
                    index
                ] = 1.0

            else:

                raw_probability[
                    0
                ] = 1.0

        except Exception:

            raw_probability[
                0
            ] = 1.0

    # ========================================================
    # PREDICTED LEVEL
    # ========================================================

    classes = getattr(
        model,
        "classes_",
        np.arange(
            1,
            len(raw_probability) + 1,
        ),
    )

    try:

        max_index = int(
            np.argmax(
                raw_probability
            )
        )

        predicted_class = (
            classes[max_index]
        )

        predicted_level = int(
            predicted_class
        )

    except Exception:

        predicted_level = (
            int(
                np.argmax(
                    raw_probability
                )
            )
            + 1
        )

    # ========================================================
    # CRITICAL LEVEL 5 CHECK
    # ========================================================

    critical, critical_term = (
        detect_level5_critical_rule(
            record
        )
    )

    if critical:

        predicted_level = 5

        probabilities = (
            critical_distribution()
        )

    else:

        probabilities = (
            normalize_to_five(
                raw_probability
            )
        )

        predicted_level = max(
            1,
            min(
                predicted_level,
                4,
            ),
        )

    # ========================================================
    # RISK NAME
    # ========================================================

    risk_names = config.get(
        "risk_level_names",
        {},
    )

    risk_name = get_risk_name(
        risk_names,
        predicted_level,
    )

    # ========================================================
    # EXPECTED RISK SCORE
    # ========================================================

    expected_score = float(
        np.sum(
            probabilities
            * np.arange(
                1,
                6,
            )
        )
    )

    return {
        "predicted_level": predicted_level,
        "risk_name": risk_name,
        "probabilities": probabilities,
        "expected_score": expected_score,
        "critical": critical,
        "critical_term": critical_term,
        "model_features": all_features,
    }


# ============================================================
# CONFIDENCE NOTE
# ============================================================

def confidence_note(
    probabilities,
    predicted_level,
):
    """
    Generate confidence interpretation.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    confidence = float(
        probabilities[
            predicted_level - 1
        ]
    )

    if confidence >= 0.80:

        return (
            "High model confidence for the predicted risk level."
        )

    if confidence >= 0.60:

        return (
            "Moderate model confidence for the predicted risk level."
        )

    return (
        "Lower model confidence. Consider manual review "
        "alongside the original ECG report."
    )


# ============================================================
# RISK GAUGE
# ============================================================

def create_risk_gauge(
    level,
    expected_score,
):
    """
    Risk level gauge.
    """

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=expected_score,
            number={
                "valueformat": ".2f",
            },
            title={
                "text": (
                    f"Risk Score — Level {level}"
                ),
            },
            gauge={
                "axis": {
                    "range": [1, 5],
                    "dtick": 1,
                },
                "bar": {
                    "color": get_risk_color(
                        level
                    ),
                },
                "steps": [
                    {
                        "range": [1, 2],
                        "color": "#E8F5E9",
                    },
                    {
                        "range": [2, 3],
                        "color": "#F1F8E9",
                    },
                    {
                        "range": [3, 4],
                        "color": "#FFF8E1",
                    },
                    {
                        "range": [4, 5],
                        "color": "#FFF3E0",
                    },
                    {
                        "range": [5, 5.01],
                        "color": "#FFEBEE",
                    },
                ],
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# PROBABILITY CHART
# ============================================================

def create_probability_chart(
    probabilities,
    names,
):
    """
    Probability distribution across five levels.
    """

    levels = list(
        range(1, 6)
    )

    labels = [
        get_risk_name(
            names,
            level,
        )
        for level in levels
    ]

    values = (
        np.asarray(
            probabilities
        )
        * 100
    )

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[
                f"{value:.1f}%"
                for value in values
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=(
            "Risk Level Probability Distribution"
        ),
        xaxis_title="Risk Level",
        yaxis_title="Probability (%)",
        yaxis=dict(
            range=[
                0,
                max(
                    100,
                    float(
                        values.max()
                        + 10
                    ),
                ),
            ],
        ),
        height=360,
        margin=dict(
            l=40,
            r=30,
            t=70,
            b=80,
        ),
    )

    return fig


# ============================================================
# HEART RATE GAUGE
# ============================================================

def create_heart_rate_gauge(
    hr,
):
    """
    Heart rate gauge.
    """

    if not is_valid_number(
        hr
    ):

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "Heart rate not available"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=18
            ),
        )

        fig.update_layout(
            height=280,
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
        )

        return fig

    hr = float(hr)

    display_value = max(
        40,
        min(
            hr,
            180,
        ),
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_value,
            number={
                "suffix": " BPM",
            },
            title={
                "text": "Heart Rate",
            },
            gauge={
                "axis": {
                    "range": [40, 180],
                    "dtick": 20,
                },
                "steps": [
                    {
                        "range": [40, 60],
                        "color": "#E3F2FD",
                    },
                    {
                        "range": [60, 100],
                        "color": "#E8F5E9",
                    },
                    {
                        "range": [100, 180],
                        "color": "#FFF3E0",
                    },
                ],
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# ECG DURATION CHART
# ============================================================

def create_duration_chart(
    record,
):
    """
    Display ECG time-domain parameters.
    """

    parameters = [
        "P Duration",
        "PR Interval",
        "QRS Duration",
        "QT",
        "QTc Bazett",
    ]

    labels = []

    values = []

    for parameter in parameters:

        value = record.get(
            parameter,
            np.nan,
        )

        if is_valid_number(
            value
        ):

            labels.append(
                parameter
            )

            values.append(
                float(value)
            )

    if not values:

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "No duration parameters available"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
        )

        fig.update_layout(
            height=360,
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
        )

        return fig

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[
                f"{value:.0f} ms"
                for value in values
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=(
            "ECG Time-Domain Parameters"
        ),
        xaxis_title="Parameter",
        yaxis_title="Duration (ms)",
        height=360,
        margin=dict(
            l=40,
            r=30,
            t=70,
            b=90,
        ),
    )

    return fig


# ============================================================
# ELECTRICAL AXIS CHART
# ============================================================

def create_axis_chart(
    p_axis,
    qrs_axis,
    t_axis,
):
    """
    Display P, QRS and T electrical axes
    on a circular 360-degree visualization.
    """

    axis_data = [
        (
            "P Axis",
            p_axis,
        ),
        (
            "QRS Axis",
            qrs_axis,
        ),
        (
            "T Axis",
            t_axis,
        ),
    ]

    valid_axes = [
        (
            name,
            float(value),
        )
        for name, value
        in axis_data
        if is_valid_number(value)
    ]

    # --------------------------------------------------------
    # No axis values
    # --------------------------------------------------------

    if not valid_axes:

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "Electrical axis data not available"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=18
            ),
        )

        fig.update_layout(
            height=450,
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
        )

        return fig

    # --------------------------------------------------------
    # Circle
    # --------------------------------------------------------

    circle_angles = np.linspace(
        0,
        360,
        361,
    )

    circle_x = np.cos(
        np.radians(
            circle_angles
        )
    )

    circle_y = -np.sin(
        np.radians(
            circle_angles
        )
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # Outer circle
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(
                width=2
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Horizontal reference
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                -1.15,
                1.15,
            ],
            y=[
                0,
                0,
            ],
            mode="lines",
            line=dict(
                width=1,
                dash="dash",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Vertical reference
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                0,
                0,
            ],
            y=[
                -1.15,
                1.15,
            ],
            mode="lines",
            line=dict(
                width=1,
                dash="dash",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    annotations = [
        dict(
            x=1.22,
            y=0,
            text="0°",
            showarrow=False,
            font=dict(
                size=13
            ),
        ),
        dict(
            x=0,
            y=-1.22,
            text="+90°",
            showarrow=False,
            font=dict(
                size=13
            ),
        ),
        dict(
            x=-1.22,
            y=0,
            text="±180°",
            showarrow=False,
            font=dict(
                size=13
            ),
        ),
        dict(
            x=0,
            y=1.22,
            text="-90°",
            showarrow=False,
            font=dict(
                size=13
            ),
        ),
    ]

    # --------------------------------------------------------
    # Vectors
    # --------------------------------------------------------

    for name, angle in valid_axes:

        radians = np.radians(
            angle
        )

        x = np.cos(
            radians
        )

        y = -np.sin(
            radians
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    0,
                    x,
                ],
                y=[
                    0,
                    y,
                ],
                mode="lines+markers",
                name=name,
                line=dict(
                    width=5
                ),
                marker=dict(
                    size=9
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Axis: {angle:.1f}°"
                    "<extra></extra>"
                ),
            )
        )

        label_x = (
            x * 1.13
        )

        label_y = (
            y * 1.13
        )

        annotations.append(
            dict(
                x=label_x,
                y=label_y,
                text=(
                    f"{name}<br>"
                    f"{angle:.1f}°"
                ),
                showarrow=False,
                font=dict(
                    size=13
                ),
            )
        )

    fig.update_layout(
        title=(
            "Electrical Axis — P / QRS / T"
        ),
        height=500,
        xaxis=dict(
            range=[
                -1.45,
                1.45,
            ],
            visible=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[
                -1.45,
                1.45,
            ],
            visible=False,
        ),
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=30,
            r=30,
            t=90,
            b=30,
        ),
    )

    return fig


# ============================================================
# AXIS COMPARISON CHART
# ============================================================

def create_axis_comparison_chart(
    record,
):
    """
    Line chart comparing P, QRS and T axis values.
    """

    names = [
        "P Axis",
        "QRS Axis",
        "T Axis",
    ]

    values = [
        record.get(
            "P Axis",
            np.nan,
        ),
        record.get(
            "QRS Axis",
            np.nan,
        ),
        record.get(
            "T Axis",
            np.nan,
        ),
    ]

    valid_names = []

    valid_values = []

    for name, value in zip(
        names,
        values,
    ):

        if is_valid_number(
            value
        ):

            valid_names.append(
                name
            )

            valid_values.append(
                float(value)
            )

    if not valid_values:

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "No electrical axis data available"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
        )

        fig.update_layout(
            height=350,
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
        )

        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=valid_names,
            y=valid_values,
            mode="lines+markers+text",
            text=[
                f"{value:.0f}°"
                for value in valid_values
            ],
            textposition="top center",
            line=dict(
                width=4
            ),
            marker=dict(
                size=11
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Axis: %{y:.1f}°"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_layout(
        title=(
            "P, QRS and T Axis Comparison"
        ),
        xaxis_title="Axis",
        yaxis_title="Degrees (°)",
        height=350,
        margin=dict(
            l=40,
            r=30,
            t=70,
            b=60,
        ),
    )

    return fig


# ============================================================
# PARAMETER STATUS CHART
# ============================================================

def create_parameter_status_chart(
    record,
):
    """
    Visualize parameters against their
    reference ranges.
    """

    parameters = [
        "Heart Rate",
        "P Duration",
        "PR Interval",
        "QRS Duration",
        "QTc Bazett",
    ]

    labels = []

    values = []

    lows = []

    highs = []

    for parameter in parameters:

        value = record.get(
            parameter,
            np.nan,
        )

        if (
            is_valid_number(value)
            and parameter
            in PARAMETER_NORMAL_RANGES
        ):

            low, high = (
                PARAMETER_NORMAL_RANGES[
                    parameter
                ]
            )

            labels.append(
                parameter
            )

            values.append(
                float(value)
            )

            lows.append(
                float(low)
            )

            highs.append(
                float(high)
            )

    if not values:

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "No reference-range parameters available"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
        )

        fig.update_layout(
            height=420,
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
        )

        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=highs,
            name="Upper Reference",
            opacity=0.25,
        )
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=lows,
            name="Lower Reference",
            opacity=0.25,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers+text",
            name="Measured Value",
            text=[
                f"{value:.0f}"
                for value in values
            ],
            textposition="top center",
            line=dict(
                width=4
            ),
            marker=dict(
                size=10
            ),
        )
    )

    fig.update_layout(
        title=(
            "Measured ECG Parameters vs Reference Ranges"
        ),
        xaxis_title="Parameter",
        yaxis_title="Value",
        barmode="overlay",
        height=420,
        margin=dict(
            l=50,
            r=30,
            t=75,
            b=90,
        ),
    )

    return fig


# ============================================================
# PARAMETER STATUS
# ============================================================

def parameter_status(
    parameter,
    value,
):
    """
    Compare parameter against reference range.
    """

    if not is_valid_number(
        value
    ):

        return "Not available"

    if parameter not in (
        PARAMETER_NORMAL_RANGES
    ):

        return "Available"

    low, high = (
        PARAMETER_NORMAL_RANGES[
            parameter
        ]
    )

    value = float(value)

    if low <= value <= high:

        return (
            "Within reference range"
        )

    return (
        "Outside reference range"
    )


# ============================================================
# FINDING SEVERITY
# ============================================================

def classify_finding(
    finding,
):
    """
    Classify machine finding.
    """

    finding_lower = (
        finding.lower()
    )

    critical_terms = [
        "ventricular fibrillation",
        "ventricular tachycardia",
        "torsades",
        "asystole",
        "stemi",
        "complete heart block",
        "third degree block",
        "myocardial infarction",
    ]

    warning_terms = [
        "tachycardia",
        "bradycardia",
        "atrial fibrillation",
        "atrial flutter",
        "pvc",
        "pac",
        "bundle branch",
        "prolonged qt",
        "t wave abnormality",
        "st-t abnormality",
        "lvh",
        "poor r wave progression",
        "first degree",
    ]

    for term in critical_terms:

        if term in finding_lower:

            return "critical"

    for term in warning_terms:

        if term in finding_lower:

            return "warning"

    if "normal ecg" in finding_lower:

        return "normal"

    return "info"


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None


if "ecg_record" not in st.session_state:

    st.session_state.ecg_record = None


if "uploaded_filename" not in st.session_state:

    st.session_state.uploaded_filename = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "ECG Risk Assessment"
)

st.caption(
    "AI-assisted ECG screening and structured cardiac "
    "parameter analysis"
)

st.divider()


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:

    model, model_config = (
        load_model_bundle()
    )

    model_loaded = True

except Exception as exc:

    model_loaded = False

    st.error(
        "The ECG risk model could not be loaded."
    )

    st.caption(
        f"Technical detail: {exc}"
    )


# ============================================================
# ECG REPORT UPLOAD
# ============================================================

st.subheader(
    "Upload ECG Report"
)

uploaded_pdf = st.file_uploader(
    "Select an ECG report",
    type=["pdf"],
    help=(
        "Upload a PDF containing structured ECG measurements."
    ),
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_pdf is not None:

    st.caption(
        f"Selected report: {uploaded_pdf.name}"
    )

    analyze = st.button(
        "Analyze ECG Report",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if not model_loaded:

            st.error(
                "Analysis cannot start because the model "
                "is unavailable."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing ECG report..."
                ):

                    pdf_text = (
                        extract_pdf_text(
                            uploaded_pdf
                        )
                    )

                    if not pdf_text.strip():

                        raise ValueError(
                            "No readable text was found "
                            "in the ECG report."
                        )

                    record = (
                        parse_ecg_pdf_text(
                            pdf_text
                        )
                    )

                    quality_problems = (
                        detect_quality_problem(
                            record
                        )
                    )

                    prediction = (
                        predict_record(
                            record,
                            model,
                            model_config,
                        )
                    )

                    st.session_state.ecg_record = (
                        record
                    )

                    st.session_state.analysis_result = {
                        **prediction,
                        "quality_problems": (
                            quality_problems
                        ),
                    }

                    st.session_state.uploaded_filename = (
                        uploaded_pdf.name
                    )

                st.success(
                    "ECG report analyzed successfully."
                )

            except Exception as exc:

                st.session_state.ecg_record = None

                st.session_state.analysis_result = (
                    None
                )

                st.error(
                    "Unable to analyze the ECG report."
                )

                st.exception(
                    exc
                )


# ============================================================
# SHOW RESULTS
# ============================================================

result = (
    st.session_state.analysis_result
)

record = (
    st.session_state.ecg_record
)


if (
    result is not None
    and record is not None
):

    level = result[
        "predicted_level"
    ]

    risk_name = result[
        "risk_name"
    ]

    probabilities = result[
        "probabilities"
    ]

    expected_score = result[
        "expected_score"
    ]

    quality_problems = result.get(
        "quality_problems",
        [],
    )

    st.divider()

    # ========================================================
    # RISK SUMMARY
    # ========================================================

    st.header(
        "Risk Assessment"
    )

    st.metric(
        label="Predicted Risk Level",
        value=f"Level {level}",
        delta=risk_name,
    )

    if level == 1:

        st.success(
            f"Level 1 — {risk_name}"
        )

    elif level == 2:

        st.success(
            f"Level 2 — {risk_name}"
        )

    elif level == 3:

        st.warning(
            f"Level 3 — {risk_name}"
        )

    elif level == 4:

        st.warning(
            f"Level 4 — {risk_name}"
        )

    else:

        st.error(
            f"Level 5 — {risk_name}"
        )

    # ========================================================
    # CRITICAL RULE
    # ========================================================

    if result.get(
        "critical"
    ):

        critical_term = result.get(
            "critical_term",
            "critical finding",
        )

        st.error(
            "Critical finding detected: "
            f"{critical_term}. "
            "The assessment was escalated to Level 5."
        )

    # ========================================================
    # QUALITY WARNING
    # ========================================================

    if quality_problems:

        st.warning(
            "Report quality / extraction warnings"
        )

        for problem in quality_problems:

            st.write(
                f"• {problem}"
            )

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = (
        st.tabs(
            [
                "Overview",
                "ECG Parameters",
                "Findings",
                "Detailed Assessment",
            ]
        )
    )

    # ========================================================
    # TAB 1 — OVERVIEW
    # ========================================================

    with tab1:

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.subheader(
                "Risk Score"
            )

            st.plotly_chart(
                create_risk_gauge(
                    level,
                    expected_score,
                ),
                use_container_width=True,
            )

        with col2:

            st.subheader(
                "Risk Probability"
            )

            risk_names = (
                model_config.get(
                    "risk_level_names",
                    {},
                )
            )

            st.plotly_chart(
                create_probability_chart(
                    probabilities,
                    risk_names,
                ),
                use_container_width=True,
            )

        st.divider()

        # ----------------------------------------------------
        # Key measurements
        # ----------------------------------------------------

        st.subheader(
            "Key ECG Measurements"
        )

        metric_columns = st.columns(
            4
        )

        metrics = [
            (
                "Heart Rate",
                record.get(
                    "Heart Rate",
                    np.nan,
                ),
                "BPM",
            ),
            (
                "PR Interval",
                record.get(
                    "PR Interval",
                    np.nan,
                ),
                "ms",
            ),
            (
                "QRS Duration",
                record.get(
                    "QRS Duration",
                    np.nan,
                ),
                "ms",
            ),
            (
                "QTc Bazett",
                record.get(
                    "QTc Bazett",
                    np.nan,
                ),
                "ms",
            ),
        ]

        for column, (
            name,
            value,
            unit,
        ) in zip(
            metric_columns,
            metrics,
        ):

            with column:

                if is_valid_number(
                    value
                ):

                    st.metric(
                        name,
                        f"{float(value):.0f} {unit}",
                    )

                else:

                    st.metric(
                        name,
                        "N/A",
                    )

        st.divider()

        # ----------------------------------------------------
        # Additional chart 1
        # ----------------------------------------------------

        st.subheader(
            "ECG Parameter Overview"
        )

        st.plotly_chart(
            create_parameter_status_chart(
                record
            ),
            use_container_width=True,
        )

        st.divider()

        # ----------------------------------------------------
        # Additional chart 2
        # ----------------------------------------------------

        st.subheader(
            "Electrical Axis Trend"
        )

        st.plotly_chart(
            create_axis_comparison_chart(
                record
            ),
            use_container_width=True,
        )

        st.divider()

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.subheader(
            "Model Confidence"
        )

        st.info(
            confidence_note(
                probabilities,
                level,
            )
        )

    # ========================================================
    # TAB 2 — ECG PARAMETERS
    # ========================================================

    with tab2:

        # ----------------------------------------------------
        # Heart Rate
        # ----------------------------------------------------

        st.subheader(
            "Heart Rate"
        )

        st.plotly_chart(
            create_heart_rate_gauge(
                record.get(
                    "Heart Rate",
                    np.nan,
                )
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        st.subheader(
            "ECG Time-Domain Parameters"
        )

        st.plotly_chart(
            create_duration_chart(
                record
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Electrical Axis
        # ----------------------------------------------------

        st.subheader(
            "Electrical Axis"
        )

        st.plotly_chart(
            create_axis_chart(
                record.get(
                    "P Axis",
                    np.nan,
                ),
                record.get(
                    "QRS Axis",
                    np.nan,
                ),
                record.get(
                    "T Axis",
                    np.nan,
                ),
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Axis numerical summary
        # ----------------------------------------------------

        axis_col1, axis_col2, axis_col3 = (
            st.columns(3)
        )

        with axis_col1:

            p_axis_value = record.get(
                "P Axis",
                np.nan,
            )

            if is_valid_number(
                p_axis_value
            ):

                st.metric(
                    "P Axis",
                    f"{float(p_axis_value):.0f}°",
                )

            else:

                st.metric(
                    "P Axis",
                    "N/A",
                )

        with axis_col2:

            qrs_axis_value = record.get(
                "QRS Axis",
                np.nan,
            )

            if is_valid_number(
                qrs_axis_value
            ):

                st.metric(
                    "QRS Axis",
                    f"{float(qrs_axis_value):.0f}°",
                )

            else:

                st.metric(
                    "QRS Axis",
                    "N/A",
                )

        with axis_col3:

            t_axis_value = record.get(
                "T Axis",
                np.nan,
            )

            if is_valid_number(
                t_axis_value
            ):

                st.metric(
                    "T Axis",
                    f"{float(t_axis_value):.0f}°",
                )

            else:

                st.metric(
                    "T Axis",
                    "N/A",
                )

        st.divider()

        # ----------------------------------------------------
        # Parameter Details
        # ----------------------------------------------------

        st.subheader(
            "Parameter Details"
        )

        parameter_rows = []

        for parameter in [
            "Heart Rate",
            "P Duration",
            "PR Interval",
            "QRS Duration",
            "QT",
            "QTc Bazett",
            "P Axis",
            "QRS Axis",
            "T Axis",
            "RV5",
            "SV1",
        ]:

            value = record.get(
                parameter,
                np.nan,
            )

            if is_valid_number(
                value
            ):

                parameter_rows.append(
                    {
                        "Parameter": parameter,
                        "Value": round(
                            float(value),
                            2,
                        ),
                        "Reference Status": (
                            parameter_status(
                                parameter,
                                value,
                            )
                        ),
                    }
                )

            else:

                parameter_rows.append(
                    {
                        "Parameter": parameter,
                        "Value": "N/A",
                        "Reference Status": (
                            "Not available"
                        ),
                    }
                )

        parameter_df = pd.DataFrame(
            parameter_rows
        )

        st.dataframe(
            parameter_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # TAB 3 — FINDINGS
    # ========================================================

    with tab3:

        st.subheader(
            "ECG Findings"
        )

        findings = record.get(
            "Machine Findings",
            [],
        )

        if findings:

            for finding in findings:

                severity = (
                    classify_finding(
                        finding
                    )
                )

                if severity == "critical":

                    st.error(
                        finding
                    )

                elif severity == "warning":

                    st.warning(
                        finding
                    )

                elif severity == "normal":

                    st.success(
                        finding
                    )

                else:

                    st.info(
                        finding
                    )

        else:

            st.info(
                "No machine-generated ECG findings "
                "were extracted from the report."
            )

        # ----------------------------------------------------
        # Rhythm
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Rhythm"
        )

        rhythm = record.get(
            "Rhythm",
            "Unknown",
        )

        st.write(
            rhythm
        )

        # ----------------------------------------------------
        # Primary Label
        # ----------------------------------------------------

        primary_label = record.get(
            "Primary Label",
            "",
        )

        if primary_label:

            st.divider()

            st.subheader(
                "Report Interpretation"
            )

            st.write(
                primary_label
            )

    # ========================================================
    # TAB 4 — DETAILED ASSESSMENT
    # ========================================================

    with tab4:

        st.subheader(
            "Risk Assessment Details"
        )

        detail_col1, detail_col2 = (
            st.columns(2)
        )

        with detail_col1:

            st.write(
                "**Predicted Level:**",
                level,
            )

            st.write(
                "**Risk Category:**",
                risk_name,
            )

            st.write(
                "**Expected Risk Score:**",
                f"{expected_score:.2f}",
            )

        with detail_col2:

            highest_probability = float(
                np.max(
                    probabilities
                )
            )

            st.write(
                "**Highest Probability:**",
                f"{highest_probability * 100:.1f}%",
            )

            st.write(
                "**Critical Rule Triggered:**",
                "Yes"
                if result.get(
                    "critical"
                )
                else "No",
            )

        st.divider()

        st.subheader(
            "Probability Breakdown"
        )

        probability_rows = []

        risk_names = (
            model_config.get(
                "risk_level_names",
                {},
            )
        )

        for index, probability in enumerate(
            probabilities,
            start=1,
        ):

            probability_rows.append(
                {
                    "Risk Level": (
                        f"Level {index}"
                    ),
                    "Risk Category": (
                        get_risk_name(
                            risk_names,
                            index,
                        )
                    ),
                    "Probability": (
                        f"{probability * 100:.2f}%"
                    ),
                }
            )

        probability_df = pd.DataFrame(
            probability_rows
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # NOTE:
        # Extracted ECG Data section intentionally removed.
        # ----------------------------------------------------

    # ========================================================
    # EXPORT
    # ========================================================

    st.divider()

    st.subheader(
        "Export Assessment"
    )

    export_data = {
        "ECG Report": (
            st.session_state.uploaded_filename
        ),
        "Predicted Risk Level": level,
        "Risk Category": risk_name,
        "Expected Risk Score": round(
            expected_score,
            4,
        ),
        "Heart Rate": record.get(
            "Heart Rate",
            np.nan,
        ),
        "P Duration": record.get(
            "P Duration",
            np.nan,
        ),
        "PR Interval": record.get(
            "PR Interval",
            np.nan,
        ),
        "QRS Duration": record.get(
            "QRS Duration",
            np.nan,
        ),
        "QT": record.get(
            "QT",
            np.nan,
        ),
        "QTc Bazett": record.get(
            "QTc Bazett",
            np.nan,
        ),
        "P Axis": record.get(
            "P Axis",
            np.nan,
        ),
        "QRS Axis": record.get(
            "QRS Axis",
            np.nan,
        ),
        "T Axis": record.get(
            "T Axis",
            np.nan,
        ),
        "RV5": record.get(
            "RV5",
            np.nan,
        ),
        "SV1": record.get(
            "SV1",
            np.nan,
        ),
        "Rhythm": record.get(
            "Rhythm",
            "",
        ),
        "Primary Label": record.get(
            "Primary Label",
            "",
        ),
        "Machine Findings": record.get(
            "Machine Findings Text",
            "",
        ),
        "Critical Rule Triggered": (
            "Yes"
            if result.get(
                "critical"
            )
            else "No"
        ),
    }

    for index, probability in enumerate(
        probabilities,
        start=1,
    ):

        export_data[
            f"Level {index} Probability"
        ] = round(
            probability,
            6,
        )

    export_df = pd.DataFrame(
        [export_data]
    )

    csv_data = export_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Assessment CSV",
        data=csv_data,
        file_name=(
            "ecg_risk_assessment.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload an ECG report to begin the assessment."
    )

    st.markdown(
        """
        ### Assessment workflow

        1. Upload the ECG report.
        2. ECG measurements are extracted from the report.
        3. Structured parameters are passed to the trained model.
        4. The model estimates the risk level.
        5. Critical findings are checked separately.
        6. Results are presented through the dashboard.
        """
    )

 
# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Clinical decision support disclaimer: This application "
    "is intended for research and screening purposes only. "
    "It does not replace interpretation by a qualified "
    "healthcare professional, physician review, or clinical "
    "diagnosis. Any critical or uncertain result should be "
    "reviewed against the original ECG report by an appropriate "
    "medical professional."
)