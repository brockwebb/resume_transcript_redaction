import logging

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure a centralized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)

    return logger
