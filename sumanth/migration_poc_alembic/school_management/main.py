from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.api import router
import os

# Create FastAPI app
app = FastAPI(
    title="School Management System",
    description="A comprehensive CRUD system for managing students and departments",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/ui/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/ui/templates")

# Include API routes
app.include_router(router, prefix="/api")

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    try:
        from app.database import create_tables
        create_tables()
        print("🚀 Application startup completed successfully")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        # Don't fail startup, let the app run without database for now
        pass

# Root endpoint - serve the UI
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "School Management System is running"}

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables (for Docker)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"🎓 Starting School Management System on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)