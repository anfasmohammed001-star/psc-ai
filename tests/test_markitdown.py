import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../flask_app')))

from markitdown import MarkItDown
from app import app, get_job, run_pdf_generation_pipeline
from services.gemini_service import GeminiClient
from models.schemas import MCQQuestion, LanguageEnum
import database

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_markitdown_import_and_conversion():
    """Verify that MarkItDown library is correctly installed and converts content to Markdown."""
    md = MarkItDown()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write("# Hello World\nThis is a test document.")
        temp_path = f.name
        
    try:
        result = md.convert(temp_path)
        assert "Hello World" in result.text_content
        assert "test document" in result.text_content
    finally:
        os.remove(temp_path)

@patch('services.gemini_service.requests.post')
def test_markdown_extraction_api(mock_post):
    """Test that extract_questions_from_markdown correctly queries the backend and returns the response."""
    # Mock OpenRouter text completion response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "[QUESTION_START]\n**Question:** Who is the father of Malayalam?\n* A) Thunchath Ezhuthachan\n* B) Cherusseri\n* C) Kunjan Nambiar\n* D) Poonthanam\n[QUESTION_END]"
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    client = GeminiClient(api_key="mock-key")
    client._gemini_client = None
    
    output = client.extract_questions_from_markdown("# Malayalam Exam\nWho is the father of Malayalam?")
    assert "[QUESTION_START]" in output
    assert "Thunchath Ezhuthachan" in output

def test_multi_format_upload_endpoint(client):
    """Verify upload endpoint accepts non-PDF formats supported by MarkItDown."""
    formats = ['docx', 'xlsx', 'pptx', 'csv', 'txt']
    
    with patch('app.executor.submit') as mock_submit:
        for fmt in formats:
            data = {
                'file': (open(__file__, 'rb'), f'test_doc.{fmt}'),
                'language': 'English'
            }
            response = client.post('/api/upload', data=data, content_type='multipart/form-data')
            assert response.status_code == 202
            json_data = response.get_json()
            assert json_data['status'] == 'PENDING'
            assert 'job_id' in json_data
            
            # Verify job is created in db
            job = get_job(json_data['job_id'])
            assert job is not None
            assert job['status'] == 'PENDING'
            
        assert mock_submit.call_count == len(formats)

@patch('services.gemini_service.GeminiClient.extract_questions_from_markdown')
@patch('services.gemini_service.GeminiClient.generate_parallel_question')
def test_text_pipeline_processing(mock_gen, mock_extract):
    """Verify that selectable text files bypass vision OCR and process using text extraction."""
    # Mock text extraction output
    mock_extract.return_value = (
        "[QUESTION_START]\n"
        "**Question:** What is the capital of France?\n"
        "* A) Paris\n"
        "* B) London\n"
        "* C) Berlin\n"
        "* D) Rome\n"
        "[QUESTION_END]"
    )
    
    # Mock parallel generation output
    mock_gen.return_value = (
        "[QUESTION_START]\n"
        "**Question:** What is the capital of Germany?\n"
        "* A) Munich\n"
        "* B) Hamburg\n"
        "* C) Berlin\n"
        "* D) Frankfurt\n"
        "* Correct Option: C\n"
        "[QUESTION_END]"
    )
    
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write("Question: What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Rome\n")
        temp_path = f.name
        
    try:
        job_id = "test-text-pipeline-job"
        database.create_job(job_id)
        
        # Run pipeline
        run_pdf_generation_pipeline(job_id, temp_path, LanguageEnum.ENGLISH)
        
        # Verify job completed successfully
        job = get_job(job_id)
        assert job["status"] == "COMPLETED"
        assert job["current_stage"] == "Finished!"
        assert job["output_pdf_path"] is not None
        assert os.path.exists(job["output_pdf_path"])
        
        # Clean up output PDF
        if os.path.exists(job["output_pdf_path"]):
            os.remove(job["output_pdf_path"])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
