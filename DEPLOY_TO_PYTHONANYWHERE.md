# Deploy Pawa AI Backend to PythonAnywhere

## Step 1: Create PythonAnywhere Account
1. Go to https://www.pythonanywhere.com/registration/register/beginner/
2. Sign up for a FREE Beginner account
3. Remember your username (you'll need it)

## Step 2: Set Up Your Backend

### A. Open Bash Console
1. After logging in, click on "Consoles" tab
2. Click "Bash" to start a new console

### B. Clone Your Repository
```bash
git clone https://github.com/joram-b-m365group/pawa-backend.git
cd pawa-backend
```

### C. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### D. Install Requirements
```bash
pip install -r requirements.txt
```

## Step 3: Configure Web App

### A. Create Web App
1. Click on "Web" tab in PythonAnywhere dashboard
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select "Python 3.10"

### B. Configure WSGI File
1. In the "Code" section, click on the WSGI configuration file link
2. **DELETE ALL** the existing content
3. Replace with this:

```python
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
```

**IMPORTANT**: Replace `YOUR_USERNAME` with your actual PythonAnywhere username!

### C. Configure Virtual Environment
1. In the "Virtualenv" section, enter:
   ```
   /home/YOUR_USERNAME/pawa-backend/venv
   ```
2. Click the checkmark to save

### D. Set Working Directory
1. In the "Code" section, set "Source code" to:
   ```
   /home/YOUR_USERNAME/pawa-backend
   ```

## Step 4: Install ASGI Server

PythonAnywhere needs an ASGI server for FastAPI. In your Bash console:

```bash
source venv/bin/activate
pip install 'uvicorn[standard]'
```

Then update your WSGI file to use ASGI:

```python
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

# Wrap FastAPI with ASGI middleware for PythonAnywhere
from asgiref.wsgi import WsgiToAsgi
application = WsgiToAsgi(app)
```

**WAIT!** Actually, there's a better way. Change the WSGI file to this:

```python
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

# For ASGI apps like FastAPI on PythonAnywhere
application = app
```

## Step 5: Reload Web App
1. Scroll to top of "Web" tab
2. Click the big green "Reload" button
3. Wait 10-20 seconds

## Step 6: Test Your Backend

Your backend will be available at:
```
https://YOUR_USERNAME.pythonanywhere.com
```

Test endpoints:
- Health: `https://YOUR_USERNAME.pythonanywhere.com/health`
- Chat: `https://YOUR_USERNAME.pythonanywhere.com/gemini/chat` (POST)

### Test with curl:
```bash
curl https://YOUR_USERNAME.pythonanywhere.com/health

curl -X POST https://YOUR_USERNAME.pythonanywhere.com/gemini/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi","conversation_history":[],"model":"gemini-2.0-flash-exp","temperature":0.7}'
```

## Step 7: Update Frontend

Once your backend is working, update the frontend to use your new URL:

**File**: `frontend/src/components/EnhancedChatInterface.tsx`

Replace all instances of:
```typescript
https://pawa-backend.onrender.com
```

With:
```typescript
https://YOUR_USERNAME.pythonanywhere.com
```

## Troubleshooting

### If you get "ImportError" or "ModuleNotFoundError":
1. Check that virtualenv path is correct in Web tab
2. Make sure all packages are installed: `pip install -r requirements.txt`
3. Check that WSGI file has correct project path

### If you get "Application object must be callable":
Make sure your WSGI file ends with `application = app` not `application = app()`

### If you get API errors:
1. Check error log in PythonAnywhere Web tab
2. Verify GOOGLE_API_KEY is set in WSGI file
3. Test the health endpoint first

### Check Logs:
1. Go to "Web" tab
2. Click on "Error log" link at the bottom
3. Look for recent errors

## Your Backend URL

After deployment, your backend will be at:
```
https://YOUR_USERNAME.pythonanywhere.com
```

Endpoints:
- `/health` - Health check
- `/gemini/health` - Gemini API health check
- `/gemini/chat` - Chat endpoint (POST)
- `/gemini/models` - List available models

## Next Steps

1. Deploy backend following steps above
2. Test all endpoints work
3. Update frontend to use new PythonAnywhere URL
4. Remove Render deployment
5. Enjoy always-on, FREE backend with NO cold starts!
