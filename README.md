# The Multimodal RAG Inference Engine

This is a web application where you upload an image of handwriting to get a personality analysis.

## What it does
1. **Upload**: You drag and drop or select an image of a handwritten note.
2. **AI Scan**: The app uses Google's Gemini AI to scan the handwriting for specific writing traits (like baseline slant, t-bar crossing, etc.).
3. **Database Match**: It looks up those traits in your local rules database (`knowledge_base.json`).
4. **Report**: It displays a psychological summary of the writer and lists each detected feature with its meaning from reference books.

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
