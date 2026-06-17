# GraphoAgent: Validated Multimodal Vision RAG & AI Governance Pipeline

## Core Engineering Phases

### Phase 1: LLM-Powered ETL Ingestion Pipeline
* **Extraction & Transformation:** Converts historical, complex, unstructured domain documentation into a type-safe database layout using `PyMuPDF` chunking and recursive LLM reformatting layers.
* **Schema Enforcement:** Enforces rigorous data contract validation utilizing `Pydantic` to output a deterministic, rapid-indexing standard `knowledge_base.json` layout, achieving 100% downstream structural compliance.

### Phase 2: Multimodal Vision RAG Extraction
* **Zero-Shot Feature Detection:** Instructs a Vision-Language Model (`gemini-2.5-flash`) to capture visual stroke variations from raw input specimens.
* **Contextual Grounding:** Implements tight Retrieval-Augmented Generation (RAG) anchor arrays. By injecting the validated database checklist directly into the vision window context, it eliminates hallucinations and forces the model to extract *only* legally valid domain criteria mappings.

### Phase 3: Multi-Agent Ingestion & De-confliction Loop
Distributes cognitive workload across three independent text-based intelligence agents running sequentially:
1. **Taxonomy Agent:** Maps extracted handwriting features cleanly into distinct psychometric pillars (*Emotional Dynamics, Social Orientation, Cognitive Style, Drive & Execution*).
2. **Forensic Jury Agent:** Actively audits concurrent fields to capture overlapping, paradoxical human traits (e.g., strong willpower vs physical exhaustion) and resolves them based on the strength of local visual evidence notes.
3. **Scoring Synthesizer:** Calculates an explicit quantitative index ($0-100\%$) for each pillar and generates a professional executive narrative overview.

### Phase 5: LLM-as-a-Judge Quality Governance Framework
* **Independent Audit Layer:** Intercepts system metrics before front-end presentation, routing the pipeline inputs and synthesized profiles to an isolated, deterministic (`temperature: 0.0`) evaluation script.
* **Programmatic Validation:** Renders automated out-of-band metrics evaluating *Faithfulness*, *Context Alignment*, and executing an explicit *Hallucination Check* to guarantee enterprise safety.

---

## 💻 Tech Stack

* **Core Language:** Python 3.11
* **AI Orchestration:** Google GenAI SDK (`gemini-2.5-flash`)
* **Front-end Environment:** Streamlit (Custom Premium Dark Theme)
* **Data Contracts & Validation:** Pydantic v2
* **Image Processing:** Pillow (PIL)

## How to run it
1. Ensure your API key is in the `.env` file:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
2. Start the app in your terminal:
   ```bash
   streamlit run app.py
   ```
3. Open the link shown in your terminal (usually `http://localhost:8501`) in your web browser.
