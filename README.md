# AI-Powered Kerala PSC Parallel Question Generator

This repository contains the complete MCA Capstone Project implementation for generating concept-matched parallel practice questions from Kerala Public Service Commission (PSC) exam papers using the Gemini 2.0 Flash Multimodal Vision API.

---

## 📌 Project Architecture & Workflow

1.  **File Upload**: The user uploads a PSC Question Paper in PDF format (and selects the paper language).
2.  **PDF Preprocessing**: The PDF is parsed and converted to high-resolution page images (120 DPI) using `pdf2image` + Poppler. Cover pages/instructions are skipped automatically in Malayalam mode.
3.  **Vision OCR Extraction**: Each page image is sent to the Gemini 2.0 Flash Vision API, extracting structured question blocks.
4.  **Concept Analysis & Generation**: The original question is analyzed and a parallel practice question is generated to match the exact syllabus concept and difficulty tier.
5.  **Quality Control Validation**: Embeddings-based similarity checks ($0.55 \le S_s \le 0.85$), options validation, and grammar checks are performed on generated questions.
6.  **PDF Compilation**: The question pairs are compiled into HTML and rendered into an A4 study guide using WeasyPrint with Noto Sans Malayalam.

---

## 📁 Repository Structure

*   `flask_app/`
    *   `app.py`: Flask API entrypoint, background thread executor, and status endpoints.
    *   `config.py`: Environment configurations and validation.
    *   `models/schemas.py`: Pydantic validation models.
    *   `services/`
        *   `pdf_service.py`: PDF converter and batch staging.
        *   `gemini_service.py`: Gemini client, backoff retrier, caching, and prompts.
        *   `question_service.py`: Text parser and duplicate checks.
        *   `validation_service.py`: Heuristic QA and bias validation engine.
        *   `render_service.py`: WeasyPrint PDF layout engine.
    *   `templates/`: HTML5 dashboard.
    *   `static/`: CSS styling stylesheets.
*   `tests/`
    *   `test_validation.py`: pytest checks for quality heuristic validations.
    *   `test_pipeline.py`: Mock API integration tests.
    *   `adversarial_gen.py`: Intentional quality violation injector.
*   `requirements.txt`: Python package dependencies.
*   `.env.example`: Configuration variables.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
*   Python 3.10 or higher.
*   **Poppler**:
    *   *Windows*: Download the latest binary from [Github release](https://github.com/oschwartz10612/poppler-windows/releases) and extract it. Add the `bin/` directory path to your system environment `PATH` variable or set `POPPLER_PATH` in your `.env` file.
    *   *macOS*: Run `brew install poppler`.
    *   *Linux*: Run `sudo apt-get install poppler-utils`.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy `.env.example` to `.env` and fill in your Gemini API key:
```bash
cp .env.example .env
```
Edit the `.env` file:
```env
SECRET_KEY=your-custom-secret-key
GEMINI_API_KEY=AIzaSy... (your google Gemini API Key)
```

---

## 🚀 Running the Server

Start the Flask local development server:
```bash
python flask_app/app.py
```
Open `http://127.0.0.1:5000` in your web browser.

---

## 🧪 Running Automated Tests

Run the test suite to verify the Quality Validation Engine and the API pipeline:
```bash
pytest -v
```
To run specific tests:
```bash
pytest tests/test_validation.py -v
```
