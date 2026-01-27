# Add parent directory to path so kiu_drone_show package can be imported
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))