# Standard library imports
import os
import io
import logging
import uuid
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Third-party imports
from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from pydantic import BaseModel
import speech_recognition as sr
from pydub import AudioSegment

# Local application imports
from agent import workflow, create_initial_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def cleanup_job():
        while True:
            try:
                cleanup_old_sessions()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("cache cleanup job failed")
                await asyncio.sleep(60)

    cleanup_task = asyncio.create_task(cleanup_job())
    logger.info("session cleanup job started")
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        logger.info("session cleanup job stopped")


# --- FastAPI Application Setup ---
app = FastAPI(title="Calendar Assistant", version="1.0.0", lifespan=lifespan)

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
    REDIRECT_URI = os.getenv("REDIRECT_URI", "https://mogambo-calendar-assistant.onrender.com/auth/callback")
else:
    REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")

# Scopes for Google Calendar API
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/calendar'
]

# Import GlobalCache from agent.py for user sessions
from agent import GlobalCache
user_sessions = GlobalCache.instance()

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
    """Get current user session from request using InMemoryStore."""
    session_id = request.session.get('session_id')
    if session_id:
        return user_sessions.get(session_id)
    return None

def require_auth(request: Request):
    """Dependency to require authentication."""
    session = get_current_user_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session

# --- WebSocket Endpoint for Live Voice Recognition ---
# Import the WebSocket handler
from websocket_handler import handle_voice_websocket

@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice recognition."""
    session_id = websocket.query_params.get('session_id')
    await handle_voice_websocket(websocket, user_sessions, session_id)

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
        logger.exception("login initiation failed")
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
        user_sessions.set_session(session_id, {
            'credentials': credentials,
            'service': calendar_service,
            'user_info': user_info,
            'created_at': datetime.now(timezone.utc),
            'conversation_history': [], # Initialize conversation history
            'last_agent_state': {} # Initialize last agent state
        })
        
        # Store session ID in browser session
        request.session['session_id'] = session_id
        request.session['user_email'] = user_info.get('email')
        
        # Redirect to chat page
        return RedirectResponse(url="/chat", status_code=302)
        
    except Exception as e:
        logger.exception("oauth callback failed")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {e}")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, session: dict = Depends(require_auth)):
    """Serve the chat page for authenticated users."""
    user_email = session['user_info'].get('email', 'User')
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user_email": user_email
    })

@app.get("/api/chat/stream")
async def chat_stream_endpoint(
    request: Request,
    message: str,
    is_voice: bool = False,
    session: dict = Depends(require_auth)
):
    """Handle streaming chat messages with server-sent events."""
    async def event_generator():
        try:
            calendar_service = session['service']
            session_id = request.session['session_id']
            user_session_data = get_current_user_session(request) or {}
            previous_state = user_session_data.get('last_agent_state')
            
            # message and is_voice are now directly from query parameters
            # No need to extract from chat_message
            
            initial_state = create_initial_state(
                user_input=message,
                session_id=session_id,
                service=calendar_service,
                is_voice=is_voice,
                previous_state=previous_state
            )
            initial_state['conversation_history'] = user_session_data.get('conversation_history', [])
            
            # Process workflow in chunks for streaming
            current_state = initial_state
            words_buffer = []
            
            async def process_response():
                result = workflow.invoke(current_state)
                response_text = result.get('final_response_text', '')
                words = response_text.split()
                for word in words:
                    words_buffer.append(word)
                    if len(words_buffer) >= 3:  # Stream in groups of 3 words
                        yield {
                            "event": "message",
                            "data": " ".join(words_buffer)
                        }
                        words_buffer.clear()
                        await asyncio.sleep(0.1)  # Small delay for natural flow
                
                if words_buffer:  # Send remaining words
                    yield {
                        "event": "message",
                        "data": " ".join(words_buffer)
                    }
                
                # Update session state
                user_session_data['conversation_history'] = result.get('conversation_history', [])
                user_session_data['last_agent_state'] = result
                user_sessions.set_session(session_id, user_session_data)
                yield {"event": "done", "data": ""}
            
            async for event in process_response():
                yield event
                
        except Exception:
            logger.exception("chat stream failed")
            yield {
                "event": "error",
                "data": "Chat stream failed"
            }

    return EventSourceResponse(event_generator())

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_message: ChatMessage,
    session: dict = Depends(require_auth)
):
    """Handle non-streaming chat messages."""
    try:
        calendar_service = session['service']
        session_id = request.session['session_id']
        
        # Ensure user_session_data is always a dictionary
        user_session_data = get_current_user_session(request) or {}
        
        previous_state = user_session_data.get('last_agent_state')
        
        message = chat_message.message
        is_voice = chat_message.is_voice
        
        # Browser audio uploads are handled by /api/voice. If this JSON endpoint
        # receives is_voice=True, treat message as already-transcribed text.
        
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
                    user_sessions.set_session(session_id, user_session_data)
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
                logger.exception("chat workflow iteration failed", extra={"session_id": session_id})
                return ChatResponse(
                    response="I had trouble processing your request. Please try again.",
                    success=False,
                    error=str(e)
                )
        
        user_session_data['conversation_history'] = conversation_history
        user_session_data['last_agent_state'] = result # Store final state
        user_sessions.set_session(session_id, user_session_data)
        
        return ChatResponse(response=response_text, success=True)
    
    except Exception as e:
        logger.exception("chat endpoint failed")
        return ChatResponse(
            response="I'm sorry, I encountered an error while processing your request. Please try again.",
            success=False,
            error=str(e)
        )

@app.post("/api/voice")
async def voice_endpoint(
    request: Request,
    audio: UploadFile = File(...), # Expect audio as multipart file data
    session: dict = Depends(require_auth)
):
    """Handle voice input specifically (OGG/Opus)."""
    try:
        calendar_service = session['service']
        session_id = request.session['session_id']
        
        audio_bytes = await audio.read()
        
        audio_format = "webm" if "webm" in (audio.content_type or "").lower() else "ogg"
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
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
                    user_sessions.set_session(session_id, user_session_data)
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
            
            except Exception:
                logger.exception("voice workflow iteration failed", extra={"session_id": session_id})
                return JSONResponse({
                    "error_message": "I had trouble processing your request. Please try again.",
                    "transcribed_text": transcribed_text,
                    "success": False
                })
        
        user_session_data['conversation_history'] = conversation_history
        user_session_data['last_agent_state'] = result # Store final state
        user_sessions.set_session(session_id, user_session_data)
        
        return JSONResponse({
            "response": response_text,
            "transcribed_text": transcribed_text,
            "success": True
        })
    
    except Exception:
        logger.exception("voice endpoint failed")
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
    
    if session_id:
        user_sessions.delete(session_id)
    
    request.session.clear()
    
    return RedirectResponse(url="/", status_code=302)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/sessions")
async def get_active_sessions():
    """Get count of active sessions (for debugging)."""
    # GlobalCache doesn't expose internal store directly for safety
    # Instead, we'll check for specific session patterns
    sessions = {
        key: timestamp.isoformat()
        for key, (_, timestamp, _) in user_sessions._cache.items()
        if isinstance(value := user_sessions.get(key), dict) and 'user_info' in value
    }
    return {
        "active_sessions": len(sessions),
        "sessions": list(sessions.keys())
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
    logger.exception("unhandled request exception", extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# --- Session Cleanup (Optional) ---
def cleanup_old_sessions():
    """Clean up expired cache entries without logging out active users."""
    user_sessions.cleanup_expired()
    logger.info("expired cache cleanup completed")

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
        
    except Exception:
        logger.exception("calendar events fetch failed")
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
        logger.exception("calendar availability check failed")
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
        logger.error("google oauth environment variables are required")
        exit(1)
    
    if not os.getenv("SESSION_SECRET_KEY"):
        logger.warning("SESSION_SECRET_KEY is not set; using insecure development fallback")

    if not os.getenv("MISTRAL_API_KEY"):
        logger.error("MISTRAL_API_KEY is required for the agent workflow")
        exit(1)

    logger.info("starting calendar assistant api", extra={
        "google_client_id_prefix": GOOGLE_CLIENT_ID[:20],
        "redirect_uri": REDIRECT_URI,
    })
    
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
