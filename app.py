import os
import json
import streamlit as st
from PIL import Image
from google import genai
from pydantic import BaseModel, Field

def load_env(env_path=".env"):
    """Loads key-value pairs from a local .env file into os.environ."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, val = line.split("=", 1)
                        val = val.strip().strip('"').strip("'")
                        os.environ[key.strip()] = val

# Execute environment configuration load
load_env()

# Set up page configurations
st.set_page_config(
    page_title="Graphology Neural Profile Engine",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force premium dark theme variables
bg_color = "#09090b"
bg_subtle = "#0d0d11"
card_color = "#12121a"
card_hover = "#1a1a24"
border_color = "#2a2a35"
border_subtle = "#1f1f2a"
text_color = "#f4f4f6"
text_muted = "#a1a1aa"
text_dim = "#71717a"
accent_color = "#7c3aed" # Purple accent for handwriting ink vibe
shadow = "none"

# Inject Custom CSS Design System
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Outfit:wght@100..900&display=swap');

/* Hide Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

/* Global styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}}

.block-container {{
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}}

/* Brand header styling */
.brand-title {{
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: {text_color};
    letter-spacing: -0.02em;
}}
.brand-subtitle {{
    font-size: 0.85rem;
    color: {text_muted};
    margin-top: 2px;
}}

/* Card titles */
.card-title {{
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.2rem;
    font-weight: 600;
    color: {text_color};
    margin-bottom: 0.25rem;
}}
.card-subtitle {{
    font-size: 0.8rem;
    color: {text_muted};
    margin-bottom: 1.25rem;
}}

/* File uploader custom style overrides */
div[data-testid="stFileUploader"] {{
    background-color: {bg_subtle} !important;
    border: 2px dashed {border_color} !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s ease;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {accent_color}99 !important;
}}
div[data-testid="stFileUploader"] > section {{
    background-color: transparent !important;
    border: none !important;
}}

/* Executive Summary styling */
.summary-box {{
    background: linear-gradient(135deg, {card_color} 0%, {bg_subtle} 100%);
    border-left: 4px solid {accent_color};
    padding: 1.25rem 1.5rem;
    border-radius: 4px 12px 12px 4px;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.65;
    color: {text_color};
    border-top: 1px solid {border_subtle};
    border-right: 1px solid {border_subtle};
    border-bottom: 1px solid {border_subtle};
}}

/* Streamlit Expander custom styling overrides */
div[data-testid="stExpander"] {{
    background-color: {card_color} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    margin-bottom: 0.75rem !important;
    box-shadow: {shadow} !important;
}}

/* Buttons custom styling overrides */
div.stButton > button:first-child {{
    background: linear-gradient(135deg, {accent_color} 0%, {accent_color}dd 100%) !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 12px -2px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
    width: 100%;
}}
div.stButton > button:first-child:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px -2px rgba(124, 58, 237, 0.45) !important;
    filter: brightness(1.05);
}}
div.stButton > button:first-child:active {{
    transform: translateY(0px) !important;
}}


</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==============================================================================
# 1. STRUCTURAL DATA SCHEMAS
# ==============================================================================
class FeatureDetection(BaseModel):
    trait_name: str = Field(description="The exact 'trait_name' copied from the provided reference checklist.")
    variation: str = Field(description="The exact 'variation' text copied from the provided reference checklist.")
    visual_evidence: str = Field(description="A detailed analysis of where and how this specific stroke or pattern is seen in the handwriting sample.")

class AnalysisReport(BaseModel):
    detected_features: list[FeatureDetection] = Field(description="List of all graphological rules confirmed present in the writing sample.")
    executive_summary: str = Field(description="An integrated, professional, psychological profile synthesis based on the combined findings.")

# ==============================================================================
# 2. DATA UTILITIES & KNOWLEDGE INGESTION
# ==============================================================================
@st.cache_data
def load_knowledge_base(kb_path="knowledge_base.json"):
    """Loads the compiled JSON knowledge base and generates a clean tracking mapping."""
    if not os.path.exists(kb_path):
        return None, "Knowledge base file not found locally."
    
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Create a lookup index for speedy mapping: (trait_name, variation) -> list of interpretations
        lookup = {}
        checklist = []
        seen = set()
        
        for item in data:
            t_name = item.get("trait_name", "").strip()
            var = item.get("variation", "").strip()
            interp = item.get("interpretation", "").strip()
            source = item.get("source_book", "").strip()
            
            if not t_name or not var:
                continue
                
            key = (t_name, var)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append({"interpretation": interp, "source": source})
            
            if key not in seen:
                seen.add(key)
                checklist.append({"trait_name": t_name, "variation": var})
                
        return {"raw": data, "lookup": lookup, "checklist": checklist}, None
    except Exception as e:
        return None, f"Failed to parse knowledge base: {str(e)}"

# ==============================================================================
# 3. CORE APPS LAYOUT & EXECUTION
# ==============================================================================
def main():
    # Ingest Phase 1 Knowledge base
    kb, error = load_knowledge_base()
    if error:
        st.error(f"❌ Core Initializer Error: {error}")
        st.info("Ensure your 'knowledge_base.json' file from Phase One is in the same directory as this script.")
        return

    # Modern Header layout
    st.markdown(
        f'<div class="brand-title">✒️ Graphology Neural Engine</div>'
        f'<div class="brand-subtitle">Automated Handwriting Profile Synthesis & Psychological Rule Matching</div>',
        unsafe_allow_html=True
    )
            
    st.markdown("<hr style='margin: 1rem 0 2rem 0; opacity: 0.15;'>", unsafe_allow_html=True)
    
    # Load API Key from environment
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in the environment. Please configure it in your `.env` file.")
        st.stop()
    else:
        client = genai.Client(api_key=api_key)

    # Application workspace split
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown(
            f'<div class="card-title">📷 Handwriting Sample Ingestion</div>'
            f'<div class="card-subtitle">Upload a high-resolution specimen image for neural analysis.</div>',
            unsafe_allow_html=True
        )
        uploaded_image = st.file_uploader(
            "Upload an image of a handwritten note or signature (PNG, JPG, JPEG):", 
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_image:
            img = Image.open(uploaded_image)
            st.markdown('<div style="margin-top: 1rem; border-radius: 8px; overflow: hidden; border: 1px solid var(--border);">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div><div style="text-align: center; font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">Target Specimen Active</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown(
            f'<div class="card-title">🧠 Analytical Synthesis Output</div>'
            f'<div class="card-subtitle">AI-generated personality insights and verified database matches.</div>',
            unsafe_allow_html=True
        )
        
        if not uploaded_image:
            st.info("Upload a handwriting specimen file in the left pane to execute the analysis engine pipeline.")
            return
            
        if st.button("🚀 Execute Neural Analysis Pipeline", type="primary"):
            with st.spinner("Analyzing high-resolution features and running cross-reference lookups..."):
                try:
                    # System Execution Prompt with explicit grounding list context
                    system_prompt = (
                        "You are an expert handwriting analyst and forensic graphologist. "
                        "Your goal is to inspect the uploaded handwriting specimen image and find occurrences "
                        "of visual traits listed in the reference checklist below.\n\n"
                        "CRITICAL STRUCTURAL REQUIREMENT:\n"
                        "You must only return matches where the 'trait_name' and 'variation' perfectly align "
                        "with strings found within the provided reference list. Do not invent new trait terms.\n\n"
                        f"--- START REFERENCE CHECKLIST ---\n"
                        f"{json.dumps(kb['checklist'], indent=2)}\n"
                        f"--- END REFERENCE CHECKLIST ---"
                    )
                    
                    # Call the Vision system via the official modern structural config parameters
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            img, 
                            "Inspect this handwriting sample image and execute feature match mapping according to your system rule guidelines."
                        ],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": AnalysisReport,
                            "system_instruction": system_prompt,
                            "temperature": 0.1 # Low variance to enforce strict pattern mapping
                        }
                    )
                    
                    # Unpack verified Pydantic output model
                    report_data: AnalysisReport = response.parsed
                    
                    # Render Analysis Summary Section
                    st.markdown(
                        f'<div style="font-family: \'Outfit\', sans-serif; font-size: 1.15rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.75rem; color: {text_color}; display: flex; align-items: center; gap: 8px;">'
                        f'📋 Executive Psychological Profile Synthesis</div>', 
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<div class="summary-box">{report_data.executive_summary}</div>', unsafe_allow_html=True)
                    
                    # Enumerate and match detected structures back to target data rules
                    st.markdown(
                        f'<div style="font-family: \'Outfit\', sans-serif; font-size: 1.15rem; font-weight: 600; margin-top: 2rem; margin-bottom: 0.75rem; color: {text_color}; display: flex; align-items: center; gap: 8px;">'
                        f'🔍 Verified Feature Rule Matches</div>', 
                        unsafe_allow_html=True
                    )
                    
                    if not report_data.detected_features:
                        st.info("The vision network completed structural scanning, but did not match explicit parameters from your rule checklist.")
                    
                    for idx, feat in enumerate(report_data.detected_features):
                        lookup_key = (feat.trait_name.strip(), feat.variation.strip())
                        
                        with st.expander(f"Feature {idx+1}: {feat.trait_name} → {feat.variation}"):
                            st.markdown(f"**👁️ Visual Evidence Observed:** *{feat.visual_evidence}*")
                            st.markdown("#### 📚 Database Rule Interpretations:")
                            
                            # Find all interpretations from the source knowledge base mapping matching this pair
                            matched_rules = kb['lookup'].get(lookup_key, [])
                            if matched_rules:
                                for rule in matched_rules:
                                    st.markdown(f"""
                                    <div style="background-color: {bg_subtle}; border-left: 3px solid {accent_color}; padding: 0.5rem 0.75rem; border-radius: 4px; margin-top: 0.5rem;">
                                        <div style="font-size: 0.85rem; font-weight: 600; color: {text_color};">💡 Significance:</div>
                                        <div style="font-size: 0.85rem; color: {text_color}; margin-top: 2px;">{rule['interpretation']}</div>
                                        <div style="font-size: 0.72rem; color: {text_muted}; margin-top: 6px;">Attribution: {rule['source']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                # Fallback handling if string alignment drifted slightly out of bounds
                                st.markdown(f"""
                                <div style="color: #f59e0b; font-size: 0.82rem; margin-top: 0.5rem;">
                                    ⚠️ Exact database look-up missed. Scanning alternative configurations...
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Soft-match recovery loop
                                soft_matchesFound = 0
                                for (t, v), rules in kb['lookup'].items():
                                    if t.lower() in lookup_key[0].lower() and v.lower() in lookup_key[1].lower():
                                        for rule in rules:
                                            st.markdown(f"""
                                            <div style="background-color: {bg_subtle}; border-left: 3px solid #f59e0b; padding: 0.5rem 0.75rem; border-radius: 4px; margin-top: 0.5rem;">
                                                <div style="font-size: 0.85rem; font-weight: 600; color: {text_color};">💡 Significance (Soft-Matched):</div>
                                                <div style="font-size: 0.85rem; color: {text_color}; margin-top: 2px;">{rule['interpretation']}</div>
                                                <div style="font-size: 0.72rem; color: {text_muted}; margin-top: 6px;">Attribution: {rule['source']}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        soft_matchesFound += 1
                                        break
                                if soft_matchesFound == 0:
                                    st.write("No matching interpretations found in local rules engine context.")
                                    
                except Exception as e:
                    st.error(f"Pipeline Interruption: {str(e)}")

if __name__ == "__main__":
    main()