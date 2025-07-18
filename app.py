# Standard library imports
import os
import io
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union

# Third-party imports
from fastapi import FastAPI, Request, HTTPException, Depends, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from pydantic import BaseModel
import speech_recognition as sr
from pydub import AudioSegment

# Local application imports
from agent import workflow, create_initial_state # Import workflow and create_initial_state from agent2.py

# --- FastAPI Application Setup ---
app = FastAPI(title="Calendar Assistant", version="1.0.0")

# Add session middleware using secret from environment
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-key-please-change-me"), # Fallback for development
    session_cookie="calendaragent_session"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# --- Google OAuth2 Configuration ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "local")

if DEPLOY_ENV == "production":
    REDIRECT_URI = "https://mogambo-voice-ai-scheduler-124439177573.asia-south2.run.app/auth/callback"
else:
    REDIRECT_URI = "http://localhost:8000/auth/callback"

# Scopes for Google Calendar API
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/calendar'
]

# In-memory storage for user sessions (for simplicity; consider persistent storage for production)
user_sessions: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models for API ---
class ChatMessage(BaseModel):
    message: str
    is_voice: bool = False

class ChatResponse(BaseModel):
    response: str = ""
    success: bool = True
    error: Optional[str] = None
    error_message: Optional[str] = None
    transcribed_text: Optional[str] = None
    needs_clarification: bool = False # Added for clarity in voice endpoint

# --- Helper Functions ---
def get_google_flow():
    """Create a Google OAuth2 flow."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise ValueError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables are not set.")
    
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = REDIRECT_URI
    return flow

def get_calendar_service(credentials):
    """Create a Google Calendar service object."""
    return build('calendar', 'v3', credentials=credentials)

def get_current_user_session(request: Request) -> Optional[Dict[str, Any]]:
    """Get current user session from request."""
    session_id = request.session.get('session_id')
    if session_id and session_id in user_sessions:
        return user_sessions[session_id]
    return None

def require_auth(request: Request):
    """Dependency to require authentication."""
    session = get_current_user_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session

# --- WebSocket Endpoint for Live Voice Recognition ---
@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    recognizer = sr.Recognizer()
    audio_data_buffer = b"" # Use a buffer to accumulate audio chunks
    
    # Define a threshold for processing (e.g., 2 seconds of audio at 16kHz, 16-bit mono)
    # 16000 samples/sec * 2 bytes/sample * 2 seconds = 64000 bytes
    PROCESSING_THRESHOLD = 64000 

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_data_buffer += data
            
            if len(audio_data_buffer) >= PROCESSING_THRESHOLD:
                try:
                    # Attempt to process as WAV first (if client sends WAV directly)
                    try:
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data_buffer), format="wav")
                    except Exception:
                        # Fallback: try to convert from webm (common for browser WebM audio)
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data_buffer), format="webm")
                    
                    # Convert to WAV for SpeechRecognition
                    wav_io = io.BytesIO()
                    audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    
                    with sr.AudioFile(wav_io) as source:
                        audio = recognizer.record(source)
                        try:
                            text = recognizer.recognize_google(audio)
                            await websocket.send_json({"transcript": text, "success": True})
                        except sr.UnknownValueError:
                            await websocket.send_json({"transcript": "Sorry, I couldn't understand that.", "success": False})
                        except sr.RequestError as e:
                            await websocket.send_json({"transcript": f"Speech recognition error: {e}", "success": False})
                except Exception as e:
                    await websocket.send_json({"transcript": f"Audio processing error: {e}", "success": False})
                
                audio_data_buffer = b"" # Reset buffer after processing
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
async def login(request: Request):
    """Initiate Google OAuth login."""
    try:
        flow = get_google_flow()
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        # Store the state in session for verification
        request.session['oauth_state'] = state
        
        return RedirectResponse(authorization_url)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Login initiation error: {e}")
        raise HTTPException(status_code=500, detail="Could not initiate Google login.")

@app.get("/auth/callback")
async def callback(request: Request, code: str, state: str):
    """Handle OAuth callback from Google."""
    try:
        # Verify state parameter
        if state != request.session.get('oauth_state'):
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        # Exchange code for credentials
        flow = get_google_flow()
        flow.fetch_token(code=code)
        
        credentials = flow.credentials
        
        # Get user info
        user_info_service = build('oauth2', 'v2', credentials=credentials)
        user_info = user_info_service.userinfo().get().execute()
        
        # Create calendar service
        calendar_service = get_calendar_service(credentials)
        
        # Create session
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            'credentials': credentials,
            'service': calendar_service,
            'user_info': user_info,
            'created_at': datetime.now(timezone.utc),
            'conversation_history': [], # Initialize conversation history
            'last_agent_state': {} # Initialize last agent state
        }
        
        # Store session ID in browser session
        request.session['session_id'] = session_id
        request.session['user_email'] = user_info.get('email')
        
        # Redirect to chat page
        return RedirectResponse(url="/chat", status_code=302)
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {e}")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, session: dict = Depends(require_auth)):
    """Serve the chat page for authenticated users."""
    user_email = session['user_info'].get('email', 'User')
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user_email": user_email
    })

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_message: ChatMessage,
    session: dict = Depends(require_auth)
):
    """Handle chat messages (both text and voice)."""
    try:
        calendar_service = session['service']
        session_id = request.session['session_id']
        
        # Ensure user_session_data is always a dictionary
        user_session_data = get_current_user_session(request) or {}
        
        previous_state = user_session_data.get('last_agent_state')
        
        message = chat_message.message
        is_voice = chat_message.is_voice
        
        if is_voice:
            try:
                audioBlob = message # Assuming message is raw audio bytes for voice input
                audio_segment = AudioSegment.from_file(io.BytesIO(audioBlob), format="ogg") # Assuming OGG/Opus
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_io) as source:
                    audio_data = recognizer.record(source)
                    message = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return ChatResponse(response="", success=False, error="Could not understand audio")
            except sr.RequestError as e:
                return ChatResponse(response="", success=False, error=f"Speech recognition error: {e}")
            except Exception as e:
                return ChatResponse(response="", success=False, error=f"Error processing audio: {e}")
        
        conversation_history = user_session_data.get('conversation_history', [])
        
        initial_state = create_initial_state(
            user_input=message,
            session_id=session_id,
            service=calendar_service,
            is_voice=is_voice,
            previous_state=previous_state
        )
        initial_state['conversation_history'] = conversation_history
        
        # Add user message to history (before workflow invocation)
        conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        max_iterations = 3
        current_iteration = 0
        response_text = ""
        
        while current_iteration < max_iterations:
            try:
                result = workflow.invoke(initial_state)
                response_text = result.get('final_response_text', '')
                
                if response_text:
                    conversation_history.append({
                        'role': 'assistant',
                        'content': response_text,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                
                if result.get('pending_clarification'):
                    user_session_data['conversation_history'] = conversation_history
                    user_session_data['last_agent_state'] = result # Store state for clarification
                    return ChatResponse(
                        response=response_text,
                        success=True,
                        needs_clarification=True # Indicate clarification needed
                    )
                
                if not result.get('missing_required_fields'):
                    break
                
                current_iteration += 1
                initial_state = result
                initial_state['conversation_history'] = conversation_history
            
            except Exception as e:
                print(f"Workflow iteration error: {e}")
                return ChatResponse(
                    response="I had trouble processing your request. Please try again.",
                    success=False,
                    error=str(e)
                )
        
        user_session_data['conversation_history'] = conversation_history
        user_session_data['last_agent_state'] = result # Store final state
        
        return ChatResponse(response=response_text, success=True)
    
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return ChatResponse(
            response="I'm sorry, I encountered an error while processing your request. Please try again.",
            success=False,
            error=str(e)
        )

@app.post("/api/voice")
async def voice_endpoint(
    request: Request,
    audio: UploadFile = Form(...), # Expect audio as form data
    session: dict = Depends(require_auth)
):
    """Handle voice input specifically (OGG/Opus)."""
    try:
        calendar_service = session['service']
        session_id = request.session['session_id']
        
        audio_bytes = await audio.read()
        
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="ogg")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            try:
                transcribed_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return JSONResponse({
                    "error_message": "Sorry, I couldn't understand the audio. Please try speaking more clearly.",
                    "transcribed_text": "",
                    "success": False
                })
            except sr.RequestError as e:
                return JSONResponse({
                    "error_message": f"There was an error with the speech recognition service: {e}",
                    "transcribed_text": "",
                    "success": False
                })
        
        if not transcribed_text:
            return JSONResponse({
                "error_message": "No voice input was detected. Please try again.",
                "transcribed_text": "",
                "success": False
            })

        user_session_data = get_current_user_session(request) or {}
        conversation_history = user_session_data.get('conversation_history', [])
        current_context = user_session_data.get('last_agent_state', {})
        
        initial_state = create_initial_state(
            user_input=transcribed_text,
            session_id=session_id,
            service=calendar_service,
            is_voice=True,
            previous_state=current_context
        )
        initial_state['conversation_history'] = conversation_history
        
        # Add user message to history (before workflow invocation)
        conversation_history.append({
            'role': 'user',
            'content': transcribed_text,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        max_iterations = 3
        current_iteration = 0
        response_text = ""
        
        while current_iteration < max_iterations:
            try:
                result = workflow.invoke(initial_state)
                response_text = result.get('final_response_text', '')
                
                # Update conversation history with assistant response
                if response_text:
                    conversation_history.append({
                        'role': 'assistant',
                        'content': response_text,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                
                if result.get('pending_clarification'):
                    user_session_data['conversation_history'] = conversation_history
                    user_session_data['last_agent_state'] = result # Store state for clarification
                    return JSONResponse({
                        "response": response_text,
                        "transcribed_text": transcribed_text,
                        "success": True,
                        "needs_clarification": True
                    })
                
                if not result.get('missing_required_fields'):
                    break
                
                current_iteration += 1
                initial_state = result
                initial_state['conversation_history'] = conversation_history
            
            except Exception as e:
                print(f"Workflow iteration error: {e}")
                return JSONResponse({
                    "error_message": "I had trouble processing your request. Please try again.",
                    "transcribed_text": transcribed_text,
                    "success": False
                })
        
        user_session_data['conversation_history'] = conversation_history
        user_session_data['last_agent_state'] = result # Store final state
        
        return JSONResponse({
            "response": response_text,
            "transcribed_text": transcribed_text,
            "success": True
        })
    
    except Exception as e:
        print(f"Voice endpoint error: {e}")
        return JSONResponse({
            "error_message": "There was an error processing your voice input. Please try again.",
            "transcribed_text": "",
            "success": False
        })

@app.get("/api/user")
async def get_user_info(request: Request, session: dict = Depends(require_auth)):
    """Get current user information."""
    user_info = session['user_info']
    return {
        "email": user_info.get('email'),
        "name": user_info.get('name'),
        "picture": user_info.get('picture')
    }

@app.post("/logout")
async def logout(request: Request):
    """Log out the current user."""
    session_id = request.session.get('session_id')
    
    if session_id and session_id in user_sessions:
        del user_sessions[session_id]
    
    request.session.clear()
    
    return RedirectResponse(url="/", status_code=302)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/sessions")
async def get_active_sessions():
    """Get count of active sessions (for debugging)."""
    return {
        "active_sessions": len(user_sessions),
        "sessions": list(user_sessions.keys())
    }

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    if exc.status_code == 401:
        return RedirectResponse(url="/")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# --- Session Cleanup (Optional) ---
def cleanup_old_sessions():
    """Clean up sessions older than 24 hours."""
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
    
    sessions_to_remove = []
    for session_id, session_data in user_sessions.items():
        if session_data['created_at'] < cutoff_time:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del user_sessions[session_id]
    
    print(f"Cleaned up {len(sessions_to_remove)} old sessions")

# --- Calendar Management Utility Endpoints (for direct API access if needed) ---
@app.get("/api/calendar/events")
async def get_calendar_events(
    request: Request,
    time_min: str,
    time_max: str,
    session: dict = Depends(require_auth)
):
    """Get calendar events for a specific time range."""
    try:
        service = session['service']
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event['id'],
                'summary': event.get('summary', 'No title'),
                'start': event['start'].get('dateTime', event['start'].get('date')),
                'end': event['end'].get('dateTime', event['end'].get('date')),
                'description': event.get('description', '')
            })
        
        return {
            "events": formatted_events,
            "success": True
        }
        
    except Exception as e:
        print(f"Error fetching calendar events: {e}")
        return {
            "events": [],
            "success": False,
            "error": str(e)
        }

@app.get("/api/calendar/availability")
async def check_availability(
    request: Request,
    time_min: str,
    time_max: str,
    session: dict = Depends(require_auth)
):
    """Check calendar availability for a time range."""
    try:
        service = session['service']
        
        body = {
            "timeMin": time_min,
            "timeMax": time_max,
            "items": [{"id": "primary"}]
        }
        
        response = service.freebusy().query(body=body).execute()
        busy_times = response['calendars']['primary']['busy']
        
        return {
            "busy_times": busy_times,
            "is_free": len(busy_times) == 0,
            "success": True
        }
        
    except Exception as e:
        print(f"Error checking availability: {e}")
        return {
            "busy_times": [],
            "is_free": False,
            "success": False,
            "error": str(e)
        }

# --- Main Application Startup ---
if __name__ == "__main__":
    import uvicorn
    
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        print("Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables are required")
        print("Please set them in your .env file or environment")
        exit(1)
    
    if not os.getenv("SESSION_SECRET_KEY"):
        print("Warning: SESSION_SECRET_KEY environment variable is not set. Using a default, which is insecure for production.")
        print("Please set it in your .env file or environment.")

    if not os.getenv("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY environment variable is required for the agent workflow.")
        print("Please set it in your .env file or environment.")
        exit(1)

    print("Starting Calendar Assistant API...")
    print(f"Google OAuth configured with client ID: {GOOGLE_CLIENT_ID[:20]}...")
    print(f"Redirect URI: {REDIRECT_URI}")
    
    uvicorn.run(
        "app:app", # Corrected to 'app:app' as the file is named app.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )