"""
Core Flask API routes, logging, and error handling middleware.
Coordinates background PDF extraction and question generation.
"""
import logging
import os
import time
import threading
from typing import Optional
from datetime import datetime
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.utils import secure_filename
import sys
# Inject flask_app directory into sys.path to resolve imports in serverless/subfolder environments (like Vercel)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.schemas import LanguageEnum, QuestionPair

# Setup directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] (%(filename)s:%(lineno)d) - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

from database import init_db, create_job, update_job as db_update_job, get_job as db_get_job, get_all_jobs

# Initialize database schema
init_db()

# Background thread pool executor
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PSC-Worker")

def update_job(job_id: str, **kwargs):
    """Updates job metadata in the SQLite database."""
    db_update_job(job_id, **kwargs)

def get_job(job_id: str) -> Optional[dict]:
    """Retrieves job metadata from the SQLite database."""
    return db_get_job(job_id)

def run_pdf_generation_pipeline(job_id: str, filepath: str, language: LanguageEnum):
    """
    Core background processing task.
    Imports services internally to avoid circular dependencies during initialization.
    """
    logger.info(f"Starting pipeline execution for job {job_id}")
    update_job(job_id, status="PROCESSING", current_stage="Initializing converters...")
    try:
        # Lazy imports of core services inside try to catch library load errors (like WeasyPrint GObject issues on Windows)
        from services.pdf_service import PDFBatchProcessor, TemporaryFileManager
        from services.gemini_service import GeminiClient
        from services.question_service import QuestionParser, DuplicateDetector
        from services.render_service import PDFRenderService
        
        # Instantiate services
        client = GeminiClient(api_key=Config.OPENROUTER_API_KEY)
        detector = DuplicateDetector(gemini_client=client)
        
        question_pairs = []
        
        # Determine if file extension is PDF
        is_pdf = filepath.lower().endswith('.pdf')
        use_text_pipeline = False
        markdown_content = ""
        
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            logger.info(f"Attempting to convert {filepath} to Markdown using MarkItDown...")
            result = md.convert(filepath)
            markdown_content = result.text_content
            if markdown_content and len(markdown_content.strip()) > 10:
                use_text_pipeline = True
                logger.info(f"MarkItDown successfully extracted {len(markdown_content)} chars of text.")
            else:
                logger.info("MarkItDown extracted minimal or empty text content.")
        except Exception as md_err:
            logger.warning(f"MarkItDown conversion failed: {md_err}")
            
        extracted_questions = []
        
        if use_text_pipeline:
            update_job(job_id, total_pages=1, current_stage="Extracting questions from Markdown text...")
            try:
                raw_ocr_output = client.extract_questions_from_markdown(markdown_content)
                extracted_questions = QuestionParser.parse_extracted_mcqs(raw_ocr_output, language)
                logger.info(f"Job {job_id}: Extracted {len(extracted_questions)} questions from Markdown text.")
            except Exception as extract_err:
                logger.warning(f"Markdown extraction failed: {extract_err}")
                extracted_questions = []
                
            if not extracted_questions and is_pdf:
                logger.info("Markdown extraction yielded 0 questions. Falling back to Vision OCR page-by-page pipeline.")
                use_text_pipeline = False

        if not use_text_pipeline:
            if not is_pdf:
                raise ValueError("Could not extract any questions from the document text.")
                
            # Step 1: Convert PDF to images
            processor = PDFBatchProcessor()
            total_pages = processor.get_total_pages(filepath)
            update_job(job_id, total_pages=total_pages)
            
            # Determine cover page skips (PSC Malayalam papers usually skip first 2 pages)
            skip_pages = 2 if language == LanguageEnum.MALAYALAM else 0
            
            with TemporaryFileManager() as temp_dir:
                update_job(job_id, current_stage="Converting PDF pages to images...")
                image_paths = processor.convert_pdf_to_images(filepath, temp_dir, skip_pages=skip_pages)
                
                total_active_pages = len(image_paths)
                update_job(job_id, total_pages=total_active_pages)
                
                for index, img_path in enumerate(image_paths):
                    page_num = index + 1
                    logger.info(f"Job {job_id}: Processing page {page_num}/{total_active_pages}")
                    update_job(
                        job_id, 
                        current_stage=f"Extracting questions from page {page_num} of {total_active_pages}...",
                        pages_processed=index
                    )
                    
                    # Extract text using Vision API
                    raw_ocr_output = client.extract_questions_from_image(img_path)
                    
                    # Parse questions
                    page_questions = QuestionParser.parse_extracted_mcqs(raw_ocr_output, language)
                    logger.info(f"Job {job_id}: Parsed {len(page_questions)} questions from page {page_num}")
                    extracted_questions.extend(page_questions)
                    
                update_job(job_id, pages_processed=total_active_pages)

        # Common parallel question generation logic
        if not extracted_questions:
            raise ValueError("No questions could be extracted from the provided document.")
            
        total_active_questions = len(extracted_questions)
        update_job(job_id, total_pages=total_active_questions)
        
        for index, orig_q in enumerate(extracted_questions):
            q_num = index + 1
            update_job(
                job_id, 
                current_stage=f"Generating parallel question {q_num} of {total_active_questions}: '{orig_q.question_text[:30]}...'",
                pages_processed=index
            )
            
            # Call Gemini Text API for parallel generation
            original_block_str = (
                f"[QUESTION_START]\n"
                f"**Question:** {orig_q.question_text}\n"
                f"* A) {orig_q.options[0]}\n"
                f"* B) {orig_q.options[1]}\n"
                f"* C) {orig_q.options[2]}\n"
                f"* D) {orig_q.options[3]}\n"
                f"[QUESTION_END]"
            )
            
            try:
                raw_gen_output = client.generate_parallel_question(original_block_str)
                generated_questions = QuestionParser.parse_extracted_mcqs(raw_gen_output, language)
            except Exception as gen_err:
                logger.warning(f"Parallel question generation failed (skipping): {str(gen_err)[:120]}")
                generated_questions = []

            if not generated_questions:
                logger.warning(f"Could not generate parallel question for: {orig_q.question_text[:60]}")
                continue

            gen_q = generated_questions[0]
            
            # Validate similarity to verify concept relevance and prevent direct duplicate
            is_dup, sim_score = detector.is_duplicate(orig_q.question_text, gen_q.question_text)
            logger.info(f"Similarity score: {sim_score:.4f} (Is duplicate? {is_dup})")
            
            pair = QuestionPair(
                original=orig_q,
                generated=gen_q,
                similarity_score=sim_score
            )
            question_pairs.append(pair)
            
            # Quota throttling delay to respect free tier RPM limits
            time.sleep(2.0)
            
        update_job(job_id, pages_processed=total_active_questions)
        
        if not question_pairs:
            raise ValueError("No parallel questions could be generated from the extracted questions.")
            
        update_job(job_id, current_stage="Compiling study guide PDF...")
        
        # Step 4: Render to PDF using WeasyPrint
        renderer = PDFRenderService(output_dir=Config.OUTPUT_FOLDER)
        output_filename = f"PSC_Study_Guide_{job_id}.pdf"
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        output_path = renderer.compile_pdf(
            title="AI-Powered Kerala PSC Parallel Practice Guide",
            pairs=[pair.model_dump() for pair in question_pairs],
            language=language.value,
            date_str=date_str,
            filename=output_filename
        )
        
        # Job Complete
        update_job(
            job_id, 
            status="COMPLETED", 
            current_stage="Finished!", 
            output_pdf_path=output_path
        )
        logger.info(f"Job {job_id} completed successfully. Output path: {output_path}")
        
    except Exception as e:
        logger.exception(f"Job {job_id} encountered an exception.")
        update_job(job_id, status="FAILED", error_message=str(e), current_stage="Failed")
        
    finally:
        # Cleanup original upload file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up uploaded file: {filepath}")
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up uploaded file {filepath}: {str(cleanup_err)}")

@app.errorhandler(Exception)
def handle_global_exception(e):
    logger.exception("Unhandled error occurred in request: %s", str(e))
    return jsonify({
        "status": "error",
        "message": "An internal server error occurred.",
        "details": str(e)
    }), 500

@app.route("/")
def index():
    """Serves the main frontend dashboard UI."""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "kerala-psc-prep-ai"}), 200

@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400
    
    file = request.files["file"]
    language_raw = request.form.get("language", "Malayalam")
    
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
    
    try:
        language = LanguageEnum(language_raw)
    except ValueError:
        return jsonify({"error": f"Invalid language. Choose from {[l.value for l in LanguageEnum]}"}), 400
        
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.csv', '.txt', '.json'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file and file_ext in ALLOWED_EXTENSIONS:
        job_id = str(uuid4())
        filename = f"{job_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Initialize job state in the database
        create_job(job_id)
        
        # Submit task to ThreadPoolExecutor
        executor.submit(run_pdf_generation_pipeline, job_id, filepath, language)
        logger.info(f"Job {job_id} submitted for async processing. File: {filename}")
        
        return jsonify({
            "job_id": job_id,
            "status": "PENDING",
            "message": "File uploaded successfully. Processing started in the background."
        }), 202
        
    return jsonify({"error": f"Unsupported file type. Supported extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job), 200

@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """Retrieves list of all past processing jobs from the SQLite database."""
    jobs = get_all_jobs()
    return jsonify(jobs), 200

@app.route("/api/download/<job_id>", methods=["GET"])
def download_pdf(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
        
    if job["status"] != "COMPLETED":
        return jsonify({"error": "Job not completed yet."}), 400
        
    if not job["output_pdf_path"] or not os.path.exists(job["output_pdf_path"]):
        return jsonify({"error": "Output PDF file not found on disk."}), 500
        
    directory = os.path.dirname(job["output_pdf_path"])
    filename = os.path.basename(job["output_pdf_path"])
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == "__main__":
    Config.validate()
    app.run(host="0.0.0.0", port=5000, debug=False)
