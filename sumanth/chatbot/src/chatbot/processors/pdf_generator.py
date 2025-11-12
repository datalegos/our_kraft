"""
PDF generator for converting scraped content to PDF format.
"""
from pathlib import Path
from typing import Optional
from datetime import datetime

from chatbot.utils.logger import logger
from chatbot.utils.exceptions import EmbeddingError


class PDFGenerator:
    """Generate PDF files from text content."""
    
    def __init__(self):
        """Initialize PDF generator."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required PDF libraries are available."""
        try:
            import reportlab
        except ImportError:
            raise EmbeddingError(
                "reportlab is required for PDF generation. "
                "Install it with: pip install reportlab"
            )
    
    def create_pdf_from_text(
        self,
        content: str,
        output_path: str,
        title: str = "Scraped Content",
        author: str = "Chatbot Scraper"
    ) -> Path:
        """
        Create a PDF file from text content.
        
        Args:
            content: Text content to convert to PDF
            output_path: Path where PDF will be saved
            title: PDF title
            author: PDF author
        
        Returns:
            Path to created PDF file
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
        except ImportError as e:
            raise EmbeddingError(f"Error importing reportlab: {e}")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Container for PDF elements
            story = []
            
            # Define styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor='#000000',
                spaceAfter=30,
                alignment=TA_LEFT
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['BodyText'],
                fontSize=10,
                textColor='#000000',
                alignment=TA_JUSTIFY,
                spaceAfter=12,
                leading=14
            )
            
            # Add title
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Add metadata
            metadata = f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            metadata += f"<b>Author:</b> {author}<br/>"
            story.append(Paragraph(metadata, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))
            
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Clean up HTML-like tags and special characters
                para = self._clean_text(para)
                
                # Split long paragraphs if needed
                if len(para) > 2000:
                    # Split at sentence boundaries
                    sentences = para.split('. ')
                    current_para = ""
                    
                    for sentence in sentences:
                        if len(current_para) + len(sentence) < 2000:
                            current_para += sentence + ". "
                        else:
                            if current_para:
                                story.append(Paragraph(current_para, body_style))
                                story.append(Spacer(1, 0.1 * inch))
                            current_para = sentence + ". "
                    
                    if current_para:
                        story.append(Paragraph(current_para, body_style))
                        story.append(Spacer(1, 0.1 * inch))
                else:
                    story.append(Paragraph(para, body_style))
                    story.append(Spacer(1, 0.1 * inch))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating PDF: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to create PDF: {e}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for PDF generation.
        
        Args:
            text: Text to clean
        
        Returns:
            Cleaned text
        """
        # Replace common HTML entities
        replacements = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def create_pdf_from_scraped_content(
        self,
        scraped_content: str,
        url: str,
        output_dir: str = "documents"
    ) -> Path:
        """
        Create PDF from scraped website content.
        
        Args:
            scraped_content: Scraped text content
            url: Source URL
            output_dir: Output directory for PDF
        
        Returns:
            Path to created PDF file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{domain}_{timestamp}.pdf"
        output_path = output_dir / filename
        
        # Create PDF
        title = f"Content from {parsed_url.netloc}"
        return self.create_pdf_from_text(
            content=scraped_content,
            output_path=str(output_path),
            title=title,
            author="Website Scraper"
        )

