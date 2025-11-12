#!/usr/bin/env python
"""
Complete pipeline: Scrape website → Create PDF → Generate Embeddings → Run Chatbot
"""
import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from chatbot.processors.scraper import Scraper, load_scraper_config
from chatbot.processors.pdf_generator import PDFGenerator
from chatbot.processors.embeddings import create_embeddings_from_documents
from chatbot.core.app import OptimizedChatbotApp
from chatbot.utils.logger import logger
from chatbot.utils.exceptions import ScraperError, EmbeddingError, ConfigurationError
from chatbot.core.config import DOCUMENT_INPUT_PATH, SCRAPER_URL, SCRAPER_MAX_PAGES


def scrape_website(url: str, max_pages: int = 10, user_agent: str = None) -> str:
    """
    Scrape content from a website.
    
    Args:
        url: Website URL to scrape
        max_pages: Maximum number of pages to scrape
        user_agent: User agent string
    
    Returns:
        Scraped content as string
    """
    logger.info(f"Starting website scrape: {url}")
    
    user_agent = user_agent or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    scraper = Scraper(user_agent=user_agent)
    content = scraper.scrape_site(url, max_pages=max_pages)
    
    if not content or not content.strip():
        raise ScraperError(f"No content scraped from {url}")
    
    logger.info(f"Scraped {len(content)} characters from {url}")
    return content


def create_pdf_from_content(content: str, url: str, output_dir: str = "documents") -> Path:
    """
    Create PDF from scraped content.
    
    Args:
        content: Scraped text content
        url: Source URL
        output_dir: Output directory for PDF
    
    Returns:
        Path to created PDF file
    """
    logger.info("Creating PDF from scraped content...")
    
    pdf_gen = PDFGenerator()
    pdf_path = pdf_gen.create_pdf_from_scraped_content(
        scraped_content=content,
        url=url,
        output_dir=output_dir
    )
    
    logger.info(f"PDF created: {pdf_path}")
    return pdf_path


def generate_embeddings(pdf_path: Path):
    """
    Generate embeddings from PDF file.
    
    Args:
        pdf_path: Path to PDF file
    """
    logger.info(f"Generating embeddings from PDF: {pdf_path}")
    
    try:
        create_embeddings_from_documents(input_path=str(pdf_path))
        logger.info("Embeddings generated successfully")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        raise EmbeddingError(f"Failed to generate embeddings: {e}")


def run_chatbot():
    """Launch the chatbot application."""
    logger.info("Launching chatbot application...")
    
    try:
        app = OptimizedChatbotApp()
        app.launch()
    except Exception as e:
        logger.error(f"Error launching chatbot: {e}", exc_info=True)
        raise


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description="Scrape website, create PDF, generate embeddings, and run chatbot"
    )
    parser.add_argument(
        "url",
        type=str,
        nargs='?',  # Make URL optional
        default=None,
        help="Website URL to scrape (optional, uses config.yaml if not provided)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to scrape (default: from config.yaml or 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="documents",
        help="Output directory for PDF (default: documents)"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping (use existing PDF)"
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF creation (use existing content)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (use existing index)"
    )
    parser.add_argument(
        "--skip-chatbot",
        action="store_true",
        help="Skip chatbot launch (just process content)"
    )
    
    args = parser.parse_args()
    
    try:
        # Get URL from command line or config
        url = args.url or SCRAPER_URL
        if not url or not url.strip():
            raise ConfigurationError(
                "No URL provided. Please either:\n"
                "  1. Pass URL as argument: python scripts/scrape_to_llm.py https://example.com\n"
                "  2. Set URL in config.yaml: scraper.url: 'https://example.com'"
            )
        
        # Get max_pages from command line or config
        max_pages = args.max_pages if args.max_pages is not None else SCRAPER_MAX_PAGES
        
        logger.info(f"Using URL: {url}")
        logger.info(f"Max pages: {max_pages}")
        
        # Step 1: Scrape website
        if not args.skip_scrape:
            content = scrape_website(url, max_pages=max_pages)
        else:
            logger.info("Skipping scraping step")
            content = None
        
        # Step 2: Create PDF
        pdf_path = None
        if not args.skip_pdf and content:
            pdf_path = create_pdf_from_content(content, url, args.output_dir)
        elif not args.skip_pdf:
            # Try to find existing PDF
            output_dir = Path(args.output_dir)
            pdfs = list(output_dir.glob("*.pdf"))
            if pdfs:
                pdf_path = pdfs[-1]  # Use most recent
                logger.info(f"Using existing PDF: {pdf_path}")
            else:
                raise ConfigurationError("No content available and no existing PDF found")
        else:
            logger.info("Skipping PDF creation step")
            # Try to find existing PDF
            output_dir = Path(args.output_dir)
            pdfs = list(output_dir.glob("*.pdf"))
            if pdfs:
                pdf_path = pdfs[-1]
                logger.info(f"Using existing PDF: {pdf_path}")
        
        # Step 3: Generate embeddings
        if not args.skip_embeddings and pdf_path:
            generate_embeddings(pdf_path)
        else:
            logger.info("Skipping embedding generation step")
        
        # Step 4: Launch chatbot
        if not args.skip_chatbot:
            logger.info("\n" + "="*60)
            logger.info("Pipeline complete! Launching chatbot...")
            logger.info("="*60 + "\n")
            run_chatbot()
        else:
            logger.info("\n" + "="*60)
            logger.info("Pipeline complete! Chatbot launch skipped.")
            logger.info("="*60 + "\n")
            logger.info("You can now run the chatbot with:")
            logger.info("  python scripts/run_chatbot.py")
        
    except (ScraperError, EmbeddingError, ConfigurationError) as e:
        logger.error(f"Pipeline error: {e}")
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n[INFO] Pipeline interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

