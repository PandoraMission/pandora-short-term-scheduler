"""
shortschedule - Science Calendar Processing and Scheduling
"""

import os
import logging
from importlib.metadata import PackageNotFoundError, version

# Package directory
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

# Import core modules
from .parser import parse_science_calendar
from .models import ScienceCalendar, Visit, ObservationSequence
from .scheduler import ScheduleProcessor
from .writer import XMLWriter

# Define public API
__all__ = [
    'parse_science_calendar',
    'ScienceCalendar',
    'Visit', 
    'ObservationSequence',
    'ScheduleProcessor',
    'XMLWriter',
    'get_version',
    'setup_logging',
]


def get_version():
    """Get package version."""
    try:
        return version("shortschedule")
    except PackageNotFoundError:
        # Fallback for development
        return "0.1.0-dev"


__version__ = get_version()


def setup_logging(level=logging.INFO):
    """
    Setup basic logging configuration.
    
    Parameters:
    -----------
    level : int
        Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger('shortschedule')


# Default logger
logger = setup_logging()

# Package metadata
__author__ = "Tom Barclay"
__description__ = "Science Calendar Processing and Scheduling for Pandora Mission"