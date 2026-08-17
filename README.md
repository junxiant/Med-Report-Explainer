# Medical Report Explainer

A medical NLP web application powered by Flask and local Large Language Models (via LM Studio) that translates complex diagnostic reports and clinical jargon into clear, concise, patient-friendly explanations.

---

## Overview

Medical reports often contain dense terminology that creates anxiety and confusion for patients. **Medical Report Explainer** provides an interface where patients can upload `.pdf` or `.txt` reports, or paste medical text directly. The system extracts the text and uses a locally hosted LLM to identify difficult terms and produce simple, accessible explanations.

## Key Features

- **Multi-Format Document Ingestion**: Upload `.pdf` or `.txt` files with server-side extraction via PyPDF2.
- **Local LLM Inference**: Direct connection to local inference servers (e.g., LM Studio) via the OpenAI API protocol for private, on-premise processing.
- **Structured Explanations**: Standardized `Term: [term]` and `Explanation: [simple explanation]` output format.
- **Side-by-Side Interface**: Dual-column web dashboard with Axios asynchronous uploads and live response rendering.

## Tech Stack

- **Backend**: Python 3.8+, Flask, Werkzeug
- **Document Extraction**: PyPDF2
- **LLM Client**: OpenAI Python SDK (LM Studio backend)
- **Frontend**: HTML5, CSS3, JavaScript, Axios


## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/junxiant/Med-Report-Explainer.git
cd Med-Report-Explainer

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install flask werkzeug pypdf2 openai
```

## Running the App
Start LM Studio:

Load your preferred local LLM (e.g., Llama, Mistral, Gemma).

Start the local server (default: http://localhost:1234/v1 or your host IP).

Configure app.py:

Set base_url and MODEL_IDENTIFIER to match your local LM Studio configuration.
Start Flask Server:

```
python app.py
Access the dashboard at http://127.0.0.1:5000.
```

Author Jun Tan

Linkedin https://www.linkedin.com/in/junxiant/
