"""
HTML template rendering and compilation into A4 study guides via xhtml2pdf.
Pure Python PDF generation with zero external C/GObject library dependencies.
Supports automated Malayalam font caching.
"""
import os
import logging
import requests
from jinja2 import Template
from xhtml2pdf import pisa

logger = logging.getLogger(__name__)

# Cache paths for the Malayalam TrueType Font files
FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "fonts")
FONT_PATH = os.path.normpath(os.path.join(FONT_DIR, "NotoSansMalayalam-GF-Regular.ttf"))
BOLD_FONT_PATH = os.path.normpath(os.path.join(FONT_DIR, "NotoSansMalayalam-GF-Bold.ttf"))

def download_font_file(local_path: str, url: str, font_name: str) -> None:
    """Downloads a font file from the given URL and saves it to the local path if it is missing or invalid."""
    # Check if file exists and has a valid size (unsubsetted font should be >= 50KB)
    if os.path.exists(local_path) and os.path.getsize(local_path) >= 50000:
        logger.info(f"{font_name} font already cached and valid.")
        return

    # Delete if invalid/subsetted font file exists
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except Exception as e:
            logger.warning(f"Could not delete outdated font file {local_path}: {str(e)}")

    logger.info(f"Downloading {font_name} font from: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)
    logger.info(f"Successfully downloaded and cached {font_name} font.")

def ensure_fonts_downloaded() -> tuple:
    """Ensures both regular and bold Noto Sans Malayalam fonts are downloaded and cached."""
    os.makedirs(FONT_DIR, exist_ok=True)
    
    regular_url = "https://fonts.gstatic.com/s/notosansmalayalam/v29/sJoi3K5XjsSdcnzn071rL37lpAOsUThnDZIfPdbeSNzVakglNM-Qw8EaeB8Nss-_RuDNG1bAjaLhEQ.ttf"
    bold_url = "https://fonts.gstatic.com/s/notosansmalayalam/v29/sJoi3K5XjsSdcnzn071rL37lpAOsUThnDZIfPdbeSNzVakglNM-Qw8EaeB8Nss-_oefNG1bAjaLhEQ.ttf"

    try:
        download_font_file(FONT_PATH, regular_url, "Noto Sans Malayalam Regular")
    except Exception as e:
        logger.error(f"Failed to download regular Malayalam font: {str(e)}")

    try:
        download_font_file(BOLD_FONT_PATH, bold_url, "Noto Sans Malayalam Bold")
    except Exception as e:
        logger.error(f"Failed to download bold Malayalam font: {str(e)}")

    return (
        "file:///" + FONT_PATH.replace("\\", "/"),
        "file:///" + BOLD_FONT_PATH.replace("\\", "/")
    )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {
        size: a4;
        margin: 2cm;
        @frame footer {
            -pdf-frame-content: footerContent;
            bottom: 1cm;
            margin-left: 2cm;
            margin-right: 2cm;
            height: 1cm;
        }
    }
    
    body {
        font-family: 'Noto Sans Malayalam', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        line-height: 1.5;
        font-size: 10pt;
    }
    
    h1 {
        text-align: center;
        color: #2c3e50;
        font-size: 18pt;
        margin-bottom: 15px;
    }
    
    .meta-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 8px 12px;
        margin-bottom: 25px;
        font-size: 9pt;
    }
    
    .question-pair {
        margin-bottom: 20px;
        border-bottom: 1px dashed #bdc3c7;
        padding-bottom: 15px;
    }
    
    .q-box {
        margin-bottom: 8px;
        padding: 6px 10px;
        border-radius: 4px;
    }
    
    .original-q {
        border-left: 3px solid #e74c3c;
        background-color: #fdf2f2;
    }
    
    .parallel-q {
        border-left: 3px solid #2ecc71;
        background-color: #f4fbf7;
        margin-top: 8px;
    }
    
    .q-title {
        font-weight: bold;
        color: #34495e;
        margin-bottom: 4px;
    }
    
    .options-grid {
        margin-left: 10px;
        margin-top: 4px;
    }
    
    .option-item {
        font-size: 9.5pt;
        margin-bottom: 2px;
    }
    
    .ans-box {
        font-size: 9pt;
        font-weight: bold;
        color: #27ae60;
        margin-top: 4px;
    }
    
    .similarity-badge {
        font-size: 8pt;
        color: #7f8c8d;
        font-weight: normal;
    }
</style>
</head>
<body>

    <!-- Header Frame content for page numbers -->
    <div id="footerContent" style="text-align: right; font-family: 'Noto Sans Malayalam'; font-size: 8pt; color: #7f8c8d;">
        Page <pdf:pagenumber/>
    </div>

    <h1>{{ title }}</h1>
    
    <div class="meta-box">
        <strong>Total Questions:</strong> {{ pairs|length }} <br>
        <strong>Language:</strong> {{ language }} <br>
        <strong>Date Generated:</strong> {{ date_str }}
    </div>

    {% for pair in pairs %}
    <div class="question-pair">
        <div class="q-box original-q">
            <div class="q-title">Q{{ loop.index }} (Original PSC Question):</div>
            <div>{{ pair.original.question_text }}</div>
            <div class="options-grid">
                <div class="option-item">A) {{ pair.original.options[0] }}</div>
                <div class="option-item">B) {{ pair.original.options[1] }}</div>
                <div class="option-item">C) {{ pair.original.options[2] }}</div>
                <div class="option-item">D) {{ pair.original.options[3] }}</div>
            </div>
        </div>
        
        <div class="q-box parallel-q">
            <div class="q-title">
                Q{{ loop.index }} Parallel Practice Question:
                <span class="similarity-badge">(Semantic Similarity: {{ "%.2f"|format(pair.similarity_score) }})</span>
            </div>
            <div>{{ pair.generated.question_text }}</div>
            <div class="options-grid">
                <div class="option-item">A) {{ pair.generated.options[0] }}</div>
                <div class="option-item">B) {{ pair.generated.options[1] }}</div>
                <div class="option-item">C) {{ pair.generated.options[2] }}</div>
                <div class="option-item">D) {{ pair.generated.options[3] }}</div>
            </div>
            <div class="ans-box">Correct Option: {{ pair.generated.correct_option }}</div>
        </div>
    </div>
    {% endfor %}

</body>
</html>
"""

class PDFRenderService:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def compile_pdf(self, title: str, pairs: list, language: str, date_str: str, filename: str) -> str:
        """
        Compiles question pairs into styled HTML and renders a PDF guide using xhtml2pdf.
        Downloads Malayalam fonts automatically if not cached.
        """
        logger.info(f"Rendering output PDF file via xhtml2pdf: {filename}")
        
        # Pre-cache local Malayalam fonts
        font_path, bold_font_path = ensure_fonts_downloaded()
        
        # Register Malayalam fonts globally in ReportLab and xhtml2pdf
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        from xhtml2pdf import default

        try:
            pdfmetrics.registerFont(TTFont('Noto Sans Malayalam', FONT_PATH))
            pdfmetrics.registerFont(TTFont('Noto Sans Malayalam-Bold', BOLD_FONT_PATH))
            addMapping('Noto Sans Malayalam', 0, 0, 'Noto Sans Malayalam')
            addMapping('Noto Sans Malayalam', 1, 0, 'Noto Sans Malayalam-Bold')
            
            default.DEFAULT_FONT['noto sans malayalam'] = 'Noto Sans Malayalam'
            default.DEFAULT_FONT['noto sans malayalam-bold'] = 'Noto Sans Malayalam-Bold'
            logger.info("Successfully registered Malayalam & Latin subset fonts globally.")
        except Exception as e:
            logger.error(f"Failed to register Malayalam fonts: {str(e)}")
            
        template = Template(HTML_TEMPLATE)
        html_content = template.render(
            title=title,
            pairs=pairs,
            language=language,
            date_str=date_str,
            font_path=font_path,
            bold_font_path=bold_font_path
        )
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(output_path, "wb") as pdf_file:
                # Compile HTML to PDF
                pisa_status = pisa.CreatePDF(
                    src=html_content,
                    dest=pdf_file
                )
                
            if pisa_status.err:
                raise RuntimeError(f"xhtml2pdf compilation failed with error code: {pisa_status.err}")
                
            logger.info(f"Successfully generated PDF study guide at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.exception("Failed compiling PDF via xhtml2pdf.")
            raise RuntimeError(f"PDF rendering failed: {str(e)}")
