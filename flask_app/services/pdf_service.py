"""
PDF preprocessing and batch image conversion using PyMuPDF (fitz).
Provides pure Python PDF-to-image extraction with zero external binary dependencies.
Includes safe temporary directory lifecycle management.
"""
import os
import shutil
import tempfile
import logging
from typing import List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class TemporaryFileManager:
    """Context manager for staging temporary page images cleanly."""
    def __init__(self, prefix: str = "psc_pdf_"):
        self.prefix = prefix
        self.dir_path = None

    def __enter__(self) -> str:
        self.dir_path = tempfile.mkdtemp(prefix=self.prefix)
        logger.info(f"Staged temporary directory created at: {self.dir_path}")
        return self.dir_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dir_path and os.path.exists(self.dir_path):
            try:
                shutil.rmtree(self.dir_path)
                logger.info(f"Cleaned up temporary directory: {self.dir_path}")
            except Exception as e:
                logger.error(f"Failed to delete temp directory {self.dir_path}: {str(e)}")


class PDFBatchProcessor:
    """Extracts pages from a PDF and converts them to high-resolution images using PyMuPDF."""
    def __init__(self, dpi: int = 120):
        self.dpi = dpi

    def get_total_pages(self, pdf_path: str) -> int:
        """Determines total pages in a PDF file using PyMuPDF."""
        try:
            with fitz.open(pdf_path) as doc:
                return len(doc)
        except Exception as e:
            logger.error(f"Failed to fetch PDF page count: {str(e)}")
            raise ValueError(f"Failed to parse PDF metadata: {str(e)}")

    def convert_pdf_to_images(
        self, 
        pdf_path: str, 
        temp_dir: str, 
        skip_pages: int = 0
    ) -> List[str]:
        """
        Converts PDF pages to individual PNG files in the specified temp directory using PyMuPDF.
        Allows skipping introductory cover pages (useful for Malayalam PSC papers).
        """
        logger.info(f"Converting {pdf_path} to image pages via PyMuPDF...")
        image_paths = []
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                start_index = skip_pages
                if start_index >= total_pages:
                    logger.warning(f"Skip page count ({skip_pages}) exceeds total pages ({total_pages}). Defaulting to start_index = 0.")
                    start_index = 0
                    
                for idx in range(start_index, total_pages):
                    page = doc.load_page(idx)
                    # Render page to a high-resolution pixmap image
                    pix = page.get_pixmap(dpi=self.dpi)
                    img_filename = f"page_{idx + 1}.png"
                    img_path = os.path.join(temp_dir, img_filename)
                    pix.save(img_path)
                    image_paths.append(img_path)
                    
            logger.info(f"Successfully converted {len(image_paths)} pages (skipped {start_index} cover pages) using PyMuPDF.")
            return image_paths
            
        except Exception as e:
            logger.exception("Failed during PDF page to image conversion.")
            raise RuntimeError(f"PDF page conversion failed: {str(e)}")
