# AE AI Starter

A small FastAPI + HTML chat starter that asks Gemini (through LangChain) for jsx.

## Files
- `app.py` - FastAPI app and embedded UI
- `requirements.txt` - Python dependencies

## Run
1. Set your Gemini API key:
   - Windows PowerShell: `$env:GEMINI_API_KEY="your_key"`
   - macOS/Linux: `export GEMINI_API_KEY="your_key"`
2. Install deps: `pip install -r requirements.txt`
3. Start server: `uvicorn app:app --reload`
4. Open: `http://127.0.0.1:8000`

