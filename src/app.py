"""
Streamlit Web Application for Customer Support Ticket Classification.
Provides an interactive interface to predict ticket categories using the trained model.
"""

from pathlib import Path
import sys
import streamlit as st

# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.joblib"

# Ensure project root is in sys.path for module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import predict_ticket from src.predict with a direct fallback if needed
try:
    from src.predict import predict_ticket, load_trained_model
except Exception:
    import joblib
    import re

    def load_trained_model(path=MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Model not found at '{path}'. Train model first.")
        return joblib.load(path)

    def predict_ticket(ticket_description: str, pipeline=None) -> str:
        if not ticket_description or not ticket_description.strip():
            raise ValueError("Ticket description cannot be empty.")
        if pipeline is None:
            pipeline = load_trained_model(MODEL_PATH)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", ticket_description.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            raise ValueError("Ticket description contains no valid words.")
        return pipeline.predict([cleaned])[0]


# Page configuration
st.set_page_config(
    page_title="Customer Ticket Classifier",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom UI styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #CBD5E1;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #202B3A;
        border: 1.5px solid #3B82F6;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1.2rem;
        text-align: center;
    }
    .result-label {
        font-size: 0.95rem;
        color: #93C5FD;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-category {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-top: 0.2rem;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load and cache the model pipeline so it doesn't reload on every rerun."""
    return load_trained_model(MODEL_PATH)


# Category descriptions and icons
CATEGORY_INFO = {
    "Account Update": {
        "icon": "👤",
        "description": "Profile changes, billing contacts, subscription seats, or permissions."
    },
    "Application Error": {
        "icon": "⚠️",
        "description": "Software crashes, unhandled exceptions, HTTP errors, or broken UI components."
    },
    "Login Issue": {
        "icon": "🔐",
        "description": "Authentication failures, password resets, 2FA/MFA, SSO, or account lockouts."
    },
    "Performance": {
        "icon": "⚡",
        "description": "Slow response times, latency spikes, page timeouts, high CPU, or memory load."
    },
    "Report": {
        "icon": "📊",
        "description": "Report exports (CSV/PDF), scheduled analytics, metric discrepancy, or dashboards."
    }
}

SAMPLE_TICKETS = {
    "Select a pre-filled sample...": "",
    "Login Issue: 2FA Authentication Failure": "Two-factor authentication SMS code is never delivered to my registered mobile phone.",
    "Application Error: HTTP 500 on Submission": "Encountering HTTP 500 Internal Server Error when trying to submit an invoice for approval.",
    "Account Update: Billing Email Change": "Please update the primary billing contact email address to accounting@company.com.",
    "Report: Corrupted CSV Export": "Exporting the Q3 inventory report to CSV results in a corrupted file with misaligned columns.",
    "Performance: Dashboard High Latency": "The main analytics dashboard takes over 45 seconds to load during peak morning hours."
}


# Sidebar: system info and supported categories
with st.sidebar:
    st.header("About the System")
    st.write(
        "This application uses a supervised **Machine Learning pipeline** "
        "(TF-IDF Vectorizer + Linear Support Vector Machine) to automatically "
        "triage customer support tickets into the appropriate department."
    )

    st.markdown("---")
    st.subheader("Supported Categories (5)")
    for cat_name, info in CATEGORY_INFO.items():
        st.markdown(f"**{info['icon']} {cat_name}**")
        st.caption(info["description"])

    st.markdown("---")
    st.subheader("Model Information")
    st.markdown(
        """
        - **Algorithm**: Linear Support Vector Classifier (`LinearSVC`)
        - **Features**: TF-IDF (Unigrams & Bigrams)
        - **Training Data**: Reviewed Customer Tickets
        - **Probability Score**: Discrete margin classification (no fabricated probabilities)
        """
    )


# Main interface
st.markdown('<div class="main-title">Customer Support Ticket Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Classify customer support tickets instantly into operational categories '
    'using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

# Load model pipeline
model_loaded = False
pipeline = None
try:
    pipeline = get_pipeline()
    model_loaded = True
except Exception as model_err:
    st.error(
        f"⚠️ **Model Artifact Error:** Unable to load trained model from `{MODEL_PATH}`.\n\n"
        f"Details: `{model_err}`\n\n"
        "Please ensure `python src/train.py` has been executed to generate `model/trained_model.joblib`."
    )

# Pre-filled sample selector for quick testing
sample_choice = st.selectbox(
    "Load an example ticket (optional):",
    options=list(SAMPLE_TICKETS.keys()),
    index=0
)
initial_text = SAMPLE_TICKETS[sample_choice] if sample_choice else ""

# Input Form
with st.form("ticket_classification_form", clear_on_submit=False):
    ticket_input = st.text_area(
        label="Customer Support Ticket Description:",
        value=initial_text,
        placeholder="e.g., Unable to log in using my corporate email address after the recent SSO migration...",
        height=140,
        help="Paste or type the customer's support request message."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.form_submit_button(
            "Predict Category",
            type="primary",
            use_container_width=True,
            disabled=not model_loaded
        )

# Prediction and result display
if submit_button:
    if not ticket_input or not ticket_input.strip():
        st.warning("⚠️ **Input Required:** Please enter a support ticket description before predicting.")
    else:
        try:
            with st.spinner("Analyzing ticket content..."):
                predicted_category = predict_ticket(ticket_input, pipeline=pipeline)

            category_meta = CATEGORY_INFO.get(predicted_category, {"icon": "📌", "description": ""})

            # Display prediction card
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Predicted Ticket Category</div>
                    <div class="result-category">{category_meta['icon']} {predicted_category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Secondary technical details
            with st.expander("Inspection Details", expanded=False):
                st.markdown(f"**Submitted Ticket:** *\"{ticket_input.strip()}\"*")
                st.markdown(f"**Category Purpose:** {category_meta['description']}")
                st.markdown("**Classifier:** Scikit-learn `LinearSVC` Pipeline (TF-IDF 1-2 ngrams)")

            st.success("✅ Ticket classified successfully.")

        except ValueError as val_err:
            st.warning(f"⚠️ **Invalid Input:** {val_err}")
        except Exception as pred_err:
            st.error(f"❌ **Prediction Error:** An unexpected error occurred: {pred_err}")

# Footer
st.markdown("---")
st.caption("Customer Support Ticket Classification System | Machine Learning Assignment")
