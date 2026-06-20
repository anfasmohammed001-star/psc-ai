"""
End-to-end integration tests for the Flask API and the PDF processing pipeline.
Mocks external Gemini API calls for hermetic testing.
"""
import sys
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

# Ensure flask_app is in Python PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../flask_app')))

from app import app, get_job
from services.render_service import PDFRenderService
from models.schemas import MCQQuestion, LanguageEnum, DifficultyEnum

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check_endpoint(client):
    """Verify that the health check endpoint returns 200 OK and status is healthy."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'kerala-psc-prep-ai'

def test_pdf_rendering_service():
    """Verify that the WeasyPrint rendering service produces a PDF document without throwing exceptions."""
    with tempfile.TemporaryDirectory() as temp_out_dir:
        renderer = PDFRenderService(output_dir=temp_out_dir)
        
        # Define dummy questions
        original = MCQQuestion(
            question_text="Sample original question?",
            options=["A", "B", "C", "D"],
            correct_option="A",
            difficulty=DifficultyEnum.EASY,
            language=LanguageEnum.ENGLISH
        )
        generated = MCQQuestion(
            question_text="Sample parallel generated question?",
            options=["A", "B", "C", "D"],
            correct_option="A",
            difficulty=DifficultyEnum.EASY,
            language=LanguageEnum.ENGLISH
        )
        
        pairs = [{
            "original": original.model_dump(),
            "generated": generated.model_dump(),
            "similarity_score": 0.72
        }]
        
        out_path = renderer.compile_pdf(
            title="Test Parallel Guide",
            pairs=pairs,
            language="English",
            date_str="2026-06-03 12:00",
            filename="test_output.pdf"
        )
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

@patch('app.executor.submit')
def test_upload_endpoint_starts_background_job(mock_submit, client):
    """Verify that uploading a PDF initializes job state and submits the task to the executor."""
    data = {
        'file': (open(__file__, 'rb'), 'test_paper.pdf'),  # Uploading current test file as dummy PDF
        'language': 'English'
    }
    
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 202
    
    json_data = response.get_json()
    assert 'job_id' in json_data
    assert json_data['status'] == 'PENDING'
    
    # Verify job exists in database
    job_id = json_data['job_id']
    job = get_job(job_id)
    assert job is not None
    assert job['status'] == 'PENDING'
    assert mock_submit.called
