"""
Configuration file for SC Location Analysis Tool
Cloud deployment version
"""

from pathlib import Path

# DATA PATH - Use relative path for cloud deployment
JSON_PATH = Path("sc_app_data.json")  # File should be in same directory as app.py

# DEFAULT PARAMETERS
DEFAULT_TRAVEL_MODE = 'drive'
DEFAULT_TIME_THRESHOLD = 10
DEFAULT_NUM_FACILITIES = 3
DEFAULT_USE_NETWORK = False

# SPEED SETTINGS
DRIVING_SPEED_MPH = 30
WALKING_SPEED_MPH = 3
WALKING_SPEED_KMH = 5.0
DEFAULT_DRIVING_SPEED = 30

# Speed profiles for different road types (mph)
SPEED_PROFILES = {
    'motorway': 65,
    'trunk': 55,
    'primary': 50,
    'secondary': 40,
    'tertiary': 30,
    'residential': 25,
    'service': 15,
    'unclassified': 25
}

# FACILITY COLORS (for visualization)
FACILITY_COLORS = {
    'Church': 'purple',
    'Primary Care': 'red',
    'Grocery': 'green',
    'Other': 'gray'
}
