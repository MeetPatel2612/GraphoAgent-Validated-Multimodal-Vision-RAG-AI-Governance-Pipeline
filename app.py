import os
import json
import time
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

load_env()

# ==============================================================================
# 0. STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Graphology Neural Profile Dashboard",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Premium Dark CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
.main, .block-container, section[data-testid="stMain"] {
    background-color: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1360px !important; }

header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, div[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

.app-header {
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 60%, #6d28d9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 0.875rem;
    color: #6b6b80;
    margin-top: 4px;
    font-weight: 400;
}

.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #c4c4d4;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.section-desc {
    font-size: 0.82rem;
    color: #6b6b80;
    margin-bottom: 1.25rem;
}

/* Upload area */
div[data-testid="stFileUploader"] {
    background-color: #0f0f1a !important;
    border: 2px dashed #2a2a3d !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #7c3aed99 !important;
}
div[data-testid="stFileUploader"] > section {
    background-color: transparent !important;
    border: none !important;
}

/* Analyze button */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.5) !important;
}

/* Summary box */
.summary-card {
    background: linear-gradient(135deg, #13131f 0%, #0f0f1a 100%);
    border: 1px solid #2a2a3d;
    border-left: 4px solid #7c3aed;
    border-radius: 4px 12px 12px 4px;
    padding: 1.25rem 1.5rem;
    font-size: 0.93rem;
    line-height: 1.75;
    color: #d4d4e8;
    margin-bottom: 1.5rem;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
    color: #8888a8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #a78bfa !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    margin-bottom: 0.6rem !important;
}
div[data-testid="stExpander"]:hover {
    border-color: #2a2a3d !important;
}

/* Dividers */
hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA SCHEMAS (Vision Layer + Agentic Analysis Layer + Audit Layer)
# ==============================================================================
# Phase 2 Vision Schema
class FeatureDetection(BaseModel):
    trait_name: str = Field(description="The exact 'trait_name' copied from the provided reference checklist.")
    variation: str = Field(description="The exact 'variation' text copied from the provided reference checklist.")
    visual_evidence: str = Field(description="Detailed analysis of where and how this specific stroke or pattern is seen in the handwriting sample.")

class InitialVisionReport(BaseModel):
    detected_features: list[FeatureDetection] = Field(description="List of all graphological rules confirmed present in the writing sample.")

# Phase 3 Multi-Agent Analytics Schema
class PillarScore(BaseModel):
    pillar_name: str = Field(description="The psychological pillar name (e.g., 'Emotional Dynamics', 'Social Orientation', 'Cognitive Style', 'Drive & Execution').")
    score: int = Field(description="A calculated numeric metric score from 0 to 100 indicating the intensity or lean of this pillar.")
    justification: str = Field(description="A analytical summary explanation citing specific extracted handwriting traits that drove this specific score.")

class ConflictResolution(BaseModel):
    contradiction_identified: str = Field(description="Description of the conflicting traits or behavior patterns found (e.g., 'High Ambition vs Low Physical Energy').")
    resolution_logic: str = Field(description="The reasoning used by the engine to resolve or balance this psychological paradox based on visual evidence notes.")

class FinalPsychologicalProfile(BaseModel):
    pillar_scores: list[PillarScore] = Field(description="The metric breakdown evaluation scores across core behavioral dimensions.")
    conflicts_resolved: list[ConflictResolution] = Field(description="A diagnostic log of all behavioral contradictions discovered and resolved by the jury layer.")
    executive_summary: str = Field(description="A highly professional, integrated psychological overview combining all findings into a cohesive profile.")

# Phase 5 Quality Audit Schema
class EvaluationReport(BaseModel):
    faithfulness_score: int = Field(description="Score from 0 to 100 measuring if the final executive summary is strictly supported by the visual evidence notes without exaggerating claims.")
    alignment_score: int = Field(description="Score from 0 to 100 assessing how accurately the agents mapped raw features to standard psychological taxonomies without drifting.")
    hallucination_detected: bool = Field(description="Set to True if the synthesizer introduced external ideas, psychological traits, or findings not present in the reference data inputs.")
    structural_critique: str = Field(description="A rigorous, objective critique detailing exact areas where pipeline reasoning succeeded or suffered from semantic drift.")

# ==============================================================================
# 2. DATA UTILITIES, BACKEND AGNOSTIC HELPERS & KNOWLEDGE INGESTION
# ==============================================================================
def generate_content_with_retry(client, model, contents, config, max_retries=3, initial_delay=3):
    """
    Robust backend wrapper that executes generation queries with an exponential 
    backoff retry strategy to safeguard against network latency and rate limits.
    """
    backoff = initial_delay
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff)
            backoff *= 2

@st.cache_data
def load_knowledge_base(kb_path="knowledge_base.json"):
    """Ingests the Phase One JSON output and structures rapid indexing schemas."""
    if not os.path.exists(kb_path):
        return None, "Knowledge base file 'knowledge_base.json' not found locally."
    
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
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
# 3. HIGH-PERFORMANCE MULTI-AGENT PIPELINE ORCHESTRATION & QA AUDITING
# ==============================================================================
def execute_cognitive_agent_cascade(client, raw_vision_features: list[dict], model_name: str):
    """Executes the Phase Three agent loop using the verified visual inputs."""
    
    # --- AGENT 1: TAXONOMY & CATEGORIZATION ---
    agent1_prompt = (
        "You are an expert industrial psychologist and data taxonomist. "
        "Review this raw list of verified handwriting features and organize them into "
        "four explicit psychological dimensions: 1. Emotional Dynamics, 2. Social Orientation, "
        "3. Cognitive Style, 4. Drive & Execution. Preserve all textual definitions and metadata."
    )
    response_agent1 = generate_content_with_retry(
        client=client,
        model=model_name,
        contents=f"Raw Input Features:\n{json.dumps(raw_vision_features, indent=2)}",
        config={"system_instruction": agent1_prompt, "temperature": 0.2}
    )
    categorized_data = response_agent1.text

    # --- AGENT 2: DE-CONFLICTION JURY ---
    agent2_prompt = (
        "You are a forensic behavioral analyst specializing in contradiction resolution. "
        "Analyze the categorized graphological data. Scan for competing indicators "
        "(e.g., signs of strong willpower vs signs of immediate yielding/weakness). "
        "Review their visual evidence strings, weigh their relevance, and write a clear analysis "
        "of how these paradoxes express themselves or which behavioral lean is dominant."
    )
    response_agent2 = generate_content_with_retry(
        client=client,
        model=model_name,
        contents=f"Categorized Input Data:\n{categorized_data}",
        config={"system_instruction": agent2_prompt, "temperature": 0.2}
    )
    deconflicted_analysis = response_agent2.text

    # --- AGENT 3: SCORING & COMPLIANCE SYNTHESIZER ---
    agent3_prompt = (
        "You are a psychometric scoring engineer and professional character profiler. "
        "Ingest the categorized groupings and the de-confliction analysis to produce the final dashboard matrix. "
        "1. Map out a solid 0-100 percentage metric score for all four core pillars. "
        "2. Parse the contradictions down into structured tracking records. "
        "3. Write a high-level corporate psychological executive summary narrative. "
        "You must output your fields strictly to mirror the requested FinalPsychologicalProfile schema layout."
    )
    
    combined_context = (
        f"--- CATEGORIZED DATA ---\n{categorized_data}\n\n"
        f"--- DE-CONFLICTION ANALYSIS ---\n{deconflicted_analysis}"
    )
    
    response_agent3 = generate_content_with_retry(
        client=client,
        model=model_name,
        contents=combined_context,
        config={
            "response_mime_type": "application/json",
            "response_schema": FinalPsychologicalProfile,
            "system_instruction": agent3_prompt,
            "temperature": 0.1
        }
    )
    return response_agent3.parsed

def execute_quality_assurance_audit(client, raw_vision_features: list[dict], synthesized_profile, model_name: str):
    """Executes the Phase Five independent out-of-band automated validation gatekeeper loop (LLM-as-a-Judge)."""
    judge_system_instruction = (
        "You are an isolated Principal AI Quality Assurance Engineer and Psychometric Auditor. "
        "Your task is to perform an objective verification audit on a multi-agent system's output. "
        "You will compare the ground-truth raw inputs (vision features) against the final summary report generated "
        "by the synthesis agents. Look for leaps in logic, ungrounded metrics, or fabricated summaries. "
        "You must output your findings strictly adhering to the requested EvaluationReport schema contract."
    )

    pipeline_inputs_context = json.dumps(raw_vision_features, indent=2)
    pipeline_outputs_context = json.dumps({
        "executive_summary": getattr(synthesized_profile, "executive_summary", ""),
        "pillar_scores": [
            {"pillar_name": p.pillar_name, "score": p.score, "justification": p.justification}
            for p in getattr(synthesized_profile, "pillar_scores", [])
        ]
    }, indent=2)

    evaluation_prompt = (
        "Please audit the following pipeline execution logs for structural soundness.\n\n"
        f"--- GROUND TRUTH PIPELINE INPUTS (VISION DETECTIONS) ---\n{pipeline_inputs_context}\n\n"
        f"--- GENERATED SYSTEM OUTPUTS (AGENT SYNTHESIS) ---\n{pipeline_outputs_context}\n\n"
        "Run compliance cross-examinations and return the completed evaluation metric matrix."
    )

    response = generate_content_with_retry(
        client=client,
        model=model_name,
        contents=evaluation_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": EvaluationReport,
            "system_instruction": judge_system_instruction,
            "temperature": 0.0
        }
    )
    return response.parsed

# ==============================================================================
# 4. MASTER WORKSPACE MAIN FUNCTION
# ==============================================================================
def main():
    # Load Database Rules Layer
    kb, error = load_knowledge_base()
    if error:
        st.error(f"Could not load knowledge base: {error}")
        return

    # API key from environment only
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found. Please add it to your `.env` file and restart the app.")
        st.stop()
    client = genai.Client(api_key=api_key)

    # ---- App Header ----
    st.markdown("""
    <div class="app-header">
        <div class="app-title">✒️ Graphology Neural Engine</div>
        <div class="app-subtitle">Upload a handwriting sample &mdash; get a full psychological profile powered by multimodal AI</div>
    </div>
    """, unsafe_allow_html=True)

    MODEL_NAME = "gemini-2.5-flash"

    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">Upload Handwriting Sample</div><div class="section-desc">Supported formats: PNG, JPG, JPEG</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop image here", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Loaded sample — ready for analysis", use_container_width=True)
            
    with col2:
        st.markdown('<div class="section-title">Analysis Output</div><div class="section-desc">Results appear here after you run the pipeline.</div>', unsafe_allow_html=True)
        if not uploaded_file:
            st.info("Upload a handwriting image on the left to get started.")
            return
            
        if st.button("🚀 Analyze Handwriting", type="primary"):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # STEP 1: Vision Scan
                status_text.info("🔍 Step 1 of 4 — Scanning handwriting features...")
                progress_bar.progress(20)
                
                vision_prompt = (
                    "You are a forensic document examiner. Inspect the handwriting sample image "
                    "and match occurrences of structural rules listed inside the grounding checklist array below. "
                    "Only extract features that match the 'trait_name' and 'variation' values exactly as listed.\n\n"
                    f"--- START REFERENCE CHECKLIST ---\n{json.dumps(kb['checklist'], indent=2)}\n--- END REFERENCE CHECKLIST ---"
                )
                
                vision_response = generate_content_with_retry(
                    client=client,
                    model=MODEL_NAME,
                    contents=[img, "Extract feature configurations present inside this image structure."],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": InitialVisionReport,
                        "system_instruction": vision_prompt,
                        "temperature": 0.1
                    }
                )
                vision_report: InitialVisionReport = vision_response.parsed
                
                if not vision_report.detected_features:
                    st.error("Pipeline stopped: The vision model found zero matching criteria hooks from your local knowledge base.")
                    return

                # Enrich vision output data with the target definitions stored in the JSON lookup
                enriched_features = []
                for feat in vision_report.detected_features:
                    lookup_key = (feat.trait_name.strip(), feat.variation.strip())
                    matched_definitions = kb['lookup'].get(lookup_key, [])
                    
                    # Core definition injection
                    interpretation_text = "Analysis verified."
                    if matched_definitions:
                        interpretation_text = " | ".join([r['interpretation'] for r in matched_definitions])
                        
                    enriched_features.append({
                        "trait_name": feat.trait_name,
                        "variation": feat.variation,
                        "visual_evidence": feat.visual_evidence,
                        "interpretation": interpretation_text
                    })

                # STEP 2: Multi-Agent Ingestion Loop
                status_text.info("🧠 Step 2 of 4 — Running behavioral analysis agents...")
                progress_bar.progress(50)
                
                final_profile: FinalPsychologicalProfile = execute_cognitive_agent_cascade(
                    client, enriched_features, MODEL_NAME
                )
                
                # STEP 3: Out-of-Band Quality Audit
                status_text.info("🛡️ Step 3 of 4 — Performing automated governance & safety audit...")
                progress_bar.progress(75)
                
                eval_report: EvaluationReport = execute_quality_assurance_audit(
                    client, enriched_features, final_profile, MODEL_NAME
                )

                # STEP 4: Dashboard Construction Line
                status_text.info("📊 Step 4 of 4 — Building your profile dashboard...")
                progress_bar.progress(95)
                
                status_text.empty()
                progress_bar.empty()

                st.success("✅ Analysis complete! Your psychological profile is ready.")

                # ---- Section A: Summary ----
                st.markdown("### 📝 Personality Overview")
                st.markdown(f'<div class="summary-card">{final_profile.executive_summary}</div>', unsafe_allow_html=True)

                # ---- Section B: Scores ----
                st.markdown("### 📊 Personality Pillar Scores")
                st.caption("Each pillar is scored 0–100 based on the detected handwriting traits.")
                metric_cols = st.columns(len(final_profile.pillar_scores) or 4)
                for idx, pillar in enumerate(final_profile.pillar_scores):
                    with metric_cols[idx % 4]:
                        st.metric(label=pillar.pillar_name, value=f"{pillar.score}%")
                        st.progress(pillar.score / 100.0)
                        st.caption(pillar.justification)

                # ---- Section C: Contradictions ----
                st.markdown("### ⚖️ Personality Contradictions")
                if not final_profile.conflicts_resolved:
                    st.caption("No conflicting traits found in this sample.")
                else:
                    for conflict in final_profile.conflicts_resolved:
                        with st.expander(f"🔍 {conflict.contradiction_identified}"):
                            st.write(conflict.resolution_logic)

                # ---- Section D: AI Governance Audit Framework ----
                st.markdown("### 🛡️ AI Governance & Integrity Audit")
                st.caption("Out-of-band evaluation metrics generated deterministically by an independent quality assurance judge loop.")
                eval_cols = st.columns(3)
                with eval_cols[0]:
                    st.metric(label="Faithfulness Score", value=f"{eval_report.faithfulness_score}%")
                    st.progress(eval_report.faithfulness_score / 100.0)
                with eval_cols[1]:
                    st.metric(label="Context Alignment", value=f"{eval_report.alignment_score}%")
                    st.progress(eval_report.alignment_score / 100.0)
                with eval_cols[2]:
                    status_val = "🚨 FAILED" if eval_report.hallucination_detected else "✅ PASSED"
                    st.metric(label="Hallucination Check", value=status_val)
                    
                with st.expander("🔬 View Auditor Structural Critique Breakdown"):
                    st.write(eval_report.structural_critique)

                # ---- Section E: Raw Features ----
                st.markdown("### 🔎 Detected Handwriting Features")
                st.caption("These are the specific traits found in the image and their psychological meanings.")
                for index, element in enumerate(enriched_features):
                    with st.expander(f"{index+1}. {element['trait_name']} — {element['variation']}"):
                        st.markdown(f"**What was seen:** {element['visual_evidence']}")
                        st.markdown(f"**What it means:** {element['interpretation']}")

            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Execution Interrupted: {str(e)}")

if __name__ == "__main__":
    main()