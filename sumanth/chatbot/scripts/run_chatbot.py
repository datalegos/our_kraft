#!/usr/bin/env python
"""
Main entry point for running the chatbot application.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from chatbot.core.app import OptimizedChatbotApp
from chatbot.utils.logger import logger
from chatbot.utils.exceptions import ConfigurationError, VectorStoreError


def main():
    """Main entry point for the application."""
    try:
        app = OptimizedChatbotApp()
        app.launch()
    except (ConfigurationError, VectorStoreError) as e:
        logger.error(f"Failed to start application: {e}")
        print(f"\n[ERROR] Error: {e}\n")
        print("Please check your configuration and try again.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n[INFO] Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

