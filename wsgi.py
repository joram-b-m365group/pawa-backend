"""
WSGI configuration for PythonAnywhere deployment
"""
import sys
import os

# Add your project directory to the sys.path
project_home = '/home/YOUR_USERNAME/pawa-backend'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBzT0i4WjPexzHG-QR5RIARNLX0ZOjK8uM'

# Import the FastAPI app
from super_intelligent_endpoint import app

# PythonAnywhere needs this
application = app
