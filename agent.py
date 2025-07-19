# Standard library imports
import os
import re
import json
import threading
from datetime import datetime, timedelta, time
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, TypedDict, Union, Any, Tuple
from collections import Counter

# Third-party imports
import pytz
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END, START
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# Voice-related imports (used by app.py for TTS, but included here for completeness if agent were standalone)
import speech_recognition as sr
from pydub.playback import play
from gtts import gTTS
from pydub import AudioSegment # Used for audio manipulation, not direct playback here

# Load environment variables for agent (redundant if app.py loads, but good for standalone testing)
load_dotenv()

# --- Type Definitions ---
class AgentState(TypedDict, total=False):
    """
    Comprehensive type definition for the agent's state.
    'total=False' allows for flexible state updates where not all keys are always present.
    """
    user_input: str
    is_voice_input: bool
    session_id: str
    service: Any  # Google Calendar Service object
    final_response_text: str
    error_message: str
    conversation_history: List[Dict[str, Any]]
    collected_info: Dict[str, Any]
    intended_tool: str
    active_event_id: str
    last_successful_tool: str
    is_general_conversation: bool
    analysis_result: Any # Should be IntentAnalysis type
    context_stack: List[Dict[str, Any]]
    tool_output: str
    skip_to_response: bool
    pending_clarification: bool
    missing_required_fields: List[str]
    clarification_question: str
    last_event_matches: List[Dict[str, Any]]
    event_context: Dict[str, Any]
    clarification_context: Dict[str, Any]
    user_preferences: Dict[str, Any]
    audio_file: str
    previous_collected_info: Dict[str, Any] # For tracking changes


# --- Global Cache System ---
class GlobalCache:
    """Thread-safe global cache for optimizing performance and managing state across sessions."""
    _instance: Optional['GlobalCache'] = None
    _cache: Dict[str, Tuple[Any, datetime]] = {}
    _lock = threading.Lock()
    MAX_CACHE_AGE = timedelta(minutes=5)

    @classmethod
    def instance(cls) -> 'GlobalCache':
        """Returns the singleton instance of GlobalCache."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the cache.
        Returns None if the key is not found or the entry has expired.
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now() - timestamp < self.MAX_CACHE_AGE:
                    return value
                else:
                    del self._cache[key]  # Expired entry
            return None

    def set(self, key: str, value: Any) -> None:
        """Stores a value in the cache with the current timestamp."""
        with self._lock:
            self._cache[key] = (value, datetime.now())
            self._cleanup()

    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidates all cache entries whose keys contain the given pattern."""
        with self._lock:
            # Ensure pattern is a string before using .replace
            if not isinstance(pattern, str):
                pattern = str(pattern)
            keys_to_delete = [k for k in self._cache.keys() if pattern.replace('*', '') in k]
            for k in keys_to_delete:
                del self._cache[k]

    def _cleanup(self) -> None:
        """Removes expired cache entries. Called automatically on `set`."""
        now = datetime.now()
        expired_keys = [
            k for k, (_, t) in self._cache.items()
            if now - t > self.MAX_CACHE_AGE
        ]
        for k in expired_keys:
            del self._cache[k]

# --- LLM Wrappers with Caching ---
class CachedLLM:
    """LLM wrapper that incorporates a global cache for responses."""
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set.")
        
        self.llm = ChatMistralAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            mistral_api_key=mistral_api_key
        )
        self.cache = GlobalCache.instance()

    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """
        Invokes the LLM, first checking the cache for a valid response.
        Caches the response if not found.
        """
        # Create a hashable key from the messages for caching
        cache_key = f"llm_{hash(json.dumps(messages, sort_keys=True))}"
        cached_response = self.cache.get(cache_key)
        if cached_response:
            # print(f"DEBUG: Cache hit for LLM with key: {cache_key}")
            return cached_response

        # print(f"DEBUG: Cache miss for LLM with key: {cache_key}. Invoking LLM...")
        response = self.llm.invoke(messages)
        self.cache.set(cache_key, response)
        return response

# Initialize optimized LLMs for different tasks
llm_for_intent_and_tool = CachedLLM(
    model_name="mistral-small-latest",
    temperature=0.1,
    max_tokens=500
)

llm_for_response = CachedLLM(
    model_name="mistral-small-latest",
    temperature=0.3,
    max_tokens=300
)

# --- Pydantic Models for Structured Output ---
class ToolType(str, Enum):
    """Enumeration of available calendar tools."""
    CREATE_EVENT = "create_calendar_event"
    RETRIEVE_EVENTS = "retrieve_calendar_events"
    UPDATE_EVENT = "update_calendar_event"
    DELETE_EVENT = "delete_calendar_event"
    FIND_FREEBUSY = "find_freebusy"
    NEED_CLARIFICATION = "need_clarification"
    GENERAL_RESPONSE = "general_response"
    NONE = "none"

class IntentAnalysis(BaseModel):
    """Structured output for intent analysis from the LLM."""
    tool: ToolType = Field(description="The appropriate tool to use based on user intent.")
    complete_args: Dict[str, Any] = Field(default_factory=dict, description="Complete arguments extracted for the tool.")
    missing_fields: List[str] = Field(default_factory=list, description="List of required fields that are missing for the tool.")
    clarification_question: str = Field(default="", description="Specific question to ask the user for clarification if needed.")
    confidence: float = Field(default=0.0, description="Confidence level (0.0-1.0) of the intent analysis.")
    reasoning: str = Field(default="", description="Brief reasoning for the determined intent and arguments.")

class GeneralResponse(BaseModel):
    """Structured output for general conversational responses."""
    response: str = Field(description="The response text for general conversation.")
    should_continue: bool = Field(default=False, description="Whether the conversation should continue after this response.")
    context_preserved: bool = Field(default=True, description="Whether the current conversational context should be preserved.")

# --- Helper Functions ---
def current_time_str() -> str:
    """Returns the current time formatted for display in IST."""
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%A, %B %d, %Y at %I:%M:%S %p IST")

def parse_relative_time(time_str: str) -> Optional[datetime]:
    """
    Parses common relative time expressions (e.g., "tomorrow", "in 2 hours").
    Returns a datetime object in Asia/Kolkata timezone or None if not recognized.
    """
    time_str = time_str.lower()
    now = datetime.now(pytz.timezone('Asia/Kolkata'))

    if "now" in time_str:
        return now

    if "in" in time_str:
        try:
            quantity_match = re.search(r'(\d+)\s*(hour|hr|minute|min|day|week)s?', time_str)
            if quantity_match:
                quantity = int(quantity_match.group(1))
                unit = quantity_match.group(2)
                if "hour" in unit:
                    return now + timedelta(hours=quantity)
                elif "minute" in unit:
                    return now + timedelta(minutes=quantity)
                elif "day" in unit:
                    return now + timedelta(days=quantity)
                elif "week" in unit:
                    return now + timedelta(weeks=quantity)
        except ValueError:
            pass

    if "morning" in time_str:
        return now.replace(hour=9, minute=0, second=0, microsecond=0)
    elif "noon" in time_str:
        return now.replace(hour=12, minute=0, second=0, microsecond=0)
    elif "afternoon" in time_str:
        return now.replace(hour=14, minute=0, second=0, microsecond=0)
    elif "evening" in time_str:
        return now.replace(hour=18, minute=0, second=0, microsecond=0)

    if "tomorrow" in time_str:
        next_day = now + timedelta(days=1)
        return next_day.replace(hour=9, minute=0, second=0, microsecond=0)
    elif "next week" in time_str:
        next_week = now + timedelta(weeks=1)
        return next_week.replace(hour=9, minute=0, second=0, microsecond=0)

    return None

def extract_time_info(text: str) -> Dict[str, Any]:
    """
    Extracts time-related information (specific time, duration, time references) from text.
    """
    text = text.lower()
    info: Dict[str, Any] = {}

    time_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text)
    if time_match:
        hour, minute, ampm = time_match.groups()
        hour = int(hour)
        minute = int(minute)
        if ampm:
            if ampm.lower() == 'pm' and hour != 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0
        info['time'] = {'hour': hour, 'minute': minute}

    duration_match = re.search(r'(\d+)\s*(hour|hr|minute|min)s?', text)
    if duration_match:
        quantity, unit = duration_match.groups()
        quantity = int(quantity)
        if 'hour' in unit or 'hr' in unit:
            info['duration'] = quantity * 60  # minutes
        else:
            info['duration'] = quantity

    time_refs = {
        'morning': (9, 0), 'noon': (12, 0), 'afternoon': (14, 0), 'evening': (18, 0)
    }
    for ref, (hour, minute) in time_refs.items():
        if ref in text:
            info['time_reference'] = {'reference': ref, 'hour': hour, 'minute': minute}

    return info

def find_events_by_summary_and_date(service: Any, calendar_id: str, summary: str, date: str) -> List[Dict]:
    """
    Finds events by fuzzy matching summary and date within a specified calendar.
    """
    try:
        start_time = f"{date}T00:00:00Z"
        end_time = f"{date}T23:59:59Z"

        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])

        matches = []
        for event in events:
            event_summary = event.get('summary', '')
            ratio = fuzz.ratio(summary.lower(), event_summary.lower())
            if ratio > 80:  # Adjustable threshold for fuzzy matching
                matches.append(event)
        return matches
    except Exception as e:
        print(f"Error finding events by summary and date: {e}")
        return []

def parse_datetime_flexible(date_str: str) -> str:
    """
    Parses flexible date/time input (e.g., "today 3pm", "tomorrow morning")
    and returns an ISO 8601 formatted string suitable for Google Calendar API.
    Defaults to 9 AM if no time is specified.
    """
    try:
        base_date: datetime.date
        now = datetime.now(pytz.timezone('Asia/Kolkata'))

        if "today" in date_str.lower():
            base_date = now.date()
        elif "tomorrow" in date_str.lower():
            base_date = (now + timedelta(days=1)).date()
        elif "next week" in date_str.lower():
            base_date = (now + timedelta(weeks=1)).date()
        else:
            # Try to parse as a specific date format
            parsed = False
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]:
                try:
                    base_date = datetime.strptime(date_str.split(' at ')[0].strip(), fmt).date()
                    parsed = True
                    break
                except ValueError:
                    continue
            if not parsed:
                base_date = now.date() # Default to today if date parsing fails

        # Extract time from string
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', date_str)
        if not time_match:
            time_match = re.search(r'(\d{1,2})\s*(AM|PM|am|pm)', date_str) # e.g., "3 PM"

        hour, minute = 9, 0 # Default time
        if time_match:
            if len(time_match.groups()) == 3: # Format like HH:MM AM/PM
                h_str, m_str, ampm = time_match.groups()
                hour = int(h_str)
                minute = int(m_str)
            else: # Format like HH AM/PM
                h_str, ampm = time_match.groups()
                hour = int(h_str)
                minute = 0

            if ampm and ampm.upper() == 'PM' and hour != 12:
                hour += 12
            elif ampm and ampm.upper() == 'AM' and hour == 12:
                hour = 0

        # Handle relative time references like "morning", "afternoon", "evening"
        if "morning" in date_str.lower() and not time_match:
            hour, minute = 9, 0
        elif "afternoon" in date_str.lower() and not time_match:
            hour, minute = 14, 0
        elif "evening" in date_str.lower() and not time_match:
            hour, minute = 18, 0

        # Combine date and time
        dt_obj = datetime.combine(base_date, time(hour, minute, 0))
        # Localize to Asia/Kolkata and then convert to ISO format
        localized_dt = pytz.timezone('Asia/Kolkata').localize(dt_obj)
        return localized_dt.isoformat()

    except Exception as e:
        print(f"Error parsing flexible datetime '{date_str}': {e}")
        # Fallback to current time if parsing fails
        return datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()

# --- Context Management ---
class ContextManager:
    """Manages and builds enhanced context for the agent's operations."""
    def __init__(self):
        self._cache = GlobalCache.instance()

    def build_enhanced_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Builds a comprehensive context dictionary for the current state,
        including conversation history, collected info, active event,
        semantic state, and user preferences. Caches the result.
        """
        session_id = state.get('session_id', 'default_session')
        user_input_hash = hash(state.get('user_input', '')) # Simple hash for input
        cache_key = f"context_{session_id}_{user_input_hash}"

        cached_context = self._cache.get(cache_key)
        if cached_context:
            return cached_context

        history = state.get('conversation_history', [])
        collected = state.get('collected_info', {})
        previous = state.get('previous_collected_info', {})
        event_ctx = state.get('event_context', {})
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)

        processed_history = []
        for msg in history[-5:]:  # Last 5 interactions
            timestamp = msg.get('timestamp', now.isoformat())
            if not timestamp.endswith('+00:00') and not timestamp.endswith('Z'):
                timestamp += '+00:00'  # Ensure UTC timezone
            try:
                msg_time = datetime.fromisoformat(timestamp)
                if msg_time.tzinfo is None:
                    msg_time = msg_time.replace(tzinfo=pytz.UTC)
                else:
                    msg_time = msg_time.astimezone(pytz.UTC)
                time_diff = now - msg_time
                if time_diff < timedelta(minutes=5):
                    temporal_marker = "just now"
                elif time_diff < timedelta(hours=1):
                    temporal_marker = f"{int(time_diff.total_seconds() / 60)} minutes ago"
                else:
                    temporal_marker = f"{int(time_diff.total_seconds() / 3600)} hours ago"
            except ValueError:
                temporal_marker = "at unknown time"

            processed_history.append(f"{msg['role']} ({temporal_marker}): {msg['content']}")

        semantic_state = {
            'time_context': {
                'current_time': now.strftime("%I:%M %p"),
                'current_date': now.strftime("%Y-%m-%d"),
                'is_working_hours': 9 <= now.hour <= 18,
                'day_of_week': now.strftime("%A"),
            },
            'interaction_state': {
                'in_clarification': bool(state.get('pending_clarification')),
                'has_active_event': bool(state.get('active_event_id')),
                'last_operation': state.get('last_successful_tool', 'none'),
                'operation_count': len(history),
            }
        }

        info_changes = {}
        if collected != previous:
            for key in set(collected.keys()) | set(previous.keys()):
                if key not in previous:
                    info_changes[key] = {'type': 'new_info', 'value': collected[key]}
                elif key not in collected:
                    info_changes[key] = {'type': 'removed_info', 'old_value': previous[key]}
                elif collected[key] != previous[key]:
                    info_changes[key] = {'type': 'modified_info', 'old_value': previous[key], 'new_value': collected[key]}

        preferences = state.get('user_preferences', {})
        if not preferences:
            event_times = []
            event_durations = []
            for msg in history:
                if msg['role'] == 'user':
                    if "morning" in msg['content'].lower(): event_times.append('morning')
                    elif "afternoon" in msg['content'].lower(): event_times.append('afternoon')
                    elif "evening" in msg['content'].lower(): event_times.append('evening')

                    duration_match = re.search(r'(\d+)\s*(hour|hr|minute|min)s?', msg['content'].lower())
                    if duration_match:
                        quantity = int(duration_match.group(1))
                        unit = duration_match.group(2)
                        if 'hour' in unit: event_durations.append(quantity * 60)
                        else: event_durations.append(quantity)

            if event_times:
                preferences['preferred_time'] = Counter(event_times).most_common(1)[0][0]
            if event_durations:
                preferences['default_duration'] = int(sum(event_durations) / len(event_durations))

        context = {
            'conversation': "\n".join(processed_history),
            'current_info': collected,
            'previous_info': previous,
            'active_event': event_ctx,
            'clarification_needed': state.get('clarification_context', {}),
            'last_tool': state.get('last_successful_tool', ''),
            'preferences': preferences,
            'semantic_state': semantic_state,
            'info_changes': info_changes,
            'error_context': {
                'has_error': bool(state.get('error_message')),
                'error_message': state.get('error_message', ''),
                'error_count': sum(1 for msg in history if 'error' in msg.get('content', '').lower())
            }
        }

        if state.get('intended_tool'):
            context['task_context'] = {
                'current_task': state['intended_tool'],
                'required_fields': state.get('missing_required_fields', []),
                'completion_status': 'pending' if state.get('missing_required_fields') else 'ready'
            }

        self._cache.set(cache_key, context)
        return context

    def update_context_stack(self, state: AgentState, context_type: str, context_data: Dict[str, Any]) -> None:
        """Updates the context stack with new information."""
        stack = state.get('context_stack', [])
        stack.append({'type': context_type, 'data': context_data, 'timestamp': datetime.now(pytz.UTC).isoformat()})
        state['context_stack'] = stack[-10:] # Keep only last 10 context entries

    def get_relevant_context(self, state: AgentState, context_type: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves relevant context based on type or returns most recent."""
        stack = state.get('context_stack', [])
        if context_type:
            for ctx in reversed(stack):
                if ctx['type'] == context_type:
                    return ctx['data']
        return stack[-1]['data'] if stack else {}

context_manager = ContextManager() # Instantiate the context manager

# --- Structured Response Helper ---
def get_structured_response(llm: CachedLLM, messages: List[Dict], response_model: BaseModel, state: AgentState) -> BaseModel:
    """
    Gets a structured response from the LLM using Pydantic validation.
    Enhances the prompt with current context and handles parsing errors gracefully.
    """
    try:
        enhanced_context = context_manager.build_enhanced_context(state)
        context_message = {"role": "system", "content": f"Current context:\n{json.dumps(enhanced_context, indent=2)}"}
        messages_with_context = [messages[0], context_message] + messages[1:] # Insert context after initial system message

        response = llm.invoke(messages_with_context)
        content = response.content if hasattr(response, 'content') else str(response)

        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

    # Use Pydantic v1 compatible method for parsing JSON
        structured_response = response_model.parse_raw(content)

        if isinstance(structured_response, IntentAnalysis):
            context_manager.update_context_stack(state, 'intent_analysis', {
                'tool': structured_response.tool.value,
                'confidence': structured_response.confidence,
                'reasoning': structured_response.reasoning
            })
        elif isinstance(structured_response, GeneralResponse):
            context_manager.update_context_stack(state, 'general_response', {
                'response': structured_response.response,
                'should_continue': structured_response.should_continue
            })

        return structured_response

    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Structured output parsing error: {e}. Raw content: {content}")
        context_manager.update_context_stack(state, 'error', {
            'type': 'parsing_error',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'raw_llm_output': content
        })

        if response_model == IntentAnalysis:
            return IntentAnalysis(
                tool=ToolType.NEED_CLARIFICATION,
                clarification_question="I couldn't fully understand your request. Could you please rephrase or provide more details?",
                reasoning=f"Parsing error: {str(e)}. LLM output was malformed or unexpected.",
                complete_args={},
                missing_fields=['rephrased_request'],
                confidence=0.3
            )
        else:
            return GeneralResponse(
                response="I'm having trouble processing that. Could you rephrase it more specifically?",
                should_continue=True,
                context_preserved=True
            )
    except Exception as e:
        print(f"An unexpected error occurred in get_structured_response: {e}")
        context_manager.update_context_stack(state, 'error', {
            'type': 'unexpected_error_structured_response',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        if response_model == IntentAnalysis:
            return IntentAnalysis(
                tool=ToolType.GENERAL_RESPONSE, # Fallback to general response
                clarification_question="I encountered an internal issue. Please try again.",
                reasoning=f"Unexpected error: {str(e)}",
                complete_args={},
                missing_fields=[],
                confidence=0.1
            )
        else:
            return GeneralResponse(
                response="An unexpected error occurred. Please try again.",
                should_continue=False,
                context_preserved=False
            )

# --- Voice I/O Helpers ---
def _play_audio_platform_agnostic(audio_file_path: str) -> None:
    """Plays an audio file using platform-specific commands."""
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return

    try:
        # Using pydub.playback.play for cross-platform compatibility
        # Requires ffplay/ffmpeg or other backend installed and in PATH
        audio = AudioSegment.from_file(audio_file_path)
        play(audio)
    except Exception as e:
        print(f"Error playing audio with pydub: {e}")
        print("Attempting fallback to system commands...")
        if os.name == 'nt':  # Windows
            os.system(f'start /min wmplayer "{audio_file_path}"')
        else:  # Unix-like systems (Linux, macOS)
            result = os.system(f'mpg123 "{audio_file_path}" 2>/dev/null')
            if result != 0:
                if os.system('which afplay >/dev/null 2>&1') == 0:
                    os.system(f'afplay "{audio_file_path}"')
                else:
                    os.system(f'ffplay -nodisp -autoexit "{audio_file_path}" 2>/dev/null')
    except Exception as e:
        print(f"Error playing audio: {e}")

# --- Node Functions ---
def detect_mode(state: AgentState) -> AgentState:
    """Determines if the input is voice or text and initializes flow control flags."""
    print("DEBUG: Entering detect_mode node.")
    state['skip_to_response'] = False
    state['is_general_conversation'] = False
    return state

def voice_detection_node(state: AgentState) -> AgentState:
    """
    Handles voice input detection and transcription using SpeechRecognition.
    Updates the state with transcribed text or an error message.
    NOTE: This node is primarily for a direct microphone input scenario.
    For web-based voice input (like in app.py), transcription happens in the endpoint.
    This node might be simplified or removed if all voice input is pre-transcribed.
    """
    if not state.get('is_voice_input', False):
        print("DEBUG: Not a voice input, skipping voice_detection_node.")
        return state

    # If user_input is already set (e.g., from app.py's voice endpoint), skip mic input
    if state.get('user_input') and state['user_input'] != "No speech detected." and \
       state['user_input'] != "Could not understand audio." and \
       state['user_input'] != "Speech recognition service error." and \
       state['user_input'] != "Unexpected voice input error.":
        print(f"DEBUG: Voice input already transcribed: \"{state['user_input']}\"")
        return state

    print("🎙️ Voice input detection started (from agent2.py node)...")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            print("🎙️ Adjusting for ambient noise, please wait...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎙️ Listening for your command...")

            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("🎙️ Processing audio...")

            transcribed_text = recognizer.recognize_google(audio, language='en-US', show_all=False)
            state['user_input'] = transcribed_text.strip()
            print(f"Transcribed: \"{state['user_input']}\"")

    except sr.WaitTimeoutError:
        state['user_input'] = "No speech detected."
        state['error_message'] = "No speech detected. Please try again."
        print("ERROR: No speech detected.")
    except sr.UnknownValueError:
        state['user_input'] = "Could not understand audio."
        state['error_message'] = "Sorry, I couldn't understand the audio. Please speak clearly."
        print("ERROR: Could not understand audio.")
    except sr.RequestError as e:
        state['user_input'] = "Speech recognition service error."
        state['error_message'] = f"Error with speech recognition service: {e}"
        print(f"ERROR: Speech recognition service error: {e}")
    except Exception as e:
        state['user_input'] = "Unexpected voice input error."
        state['error_message'] = f"Unexpected error during voice detection: {e}"
        print(f"ERROR: Unexpected voice input error: {e}")

    return state

def intelligent_intent_analysis_node(state: AgentState) -> AgentState:
    """
    Analyzes user intent using an LLM, determines the appropriate tool,
    and extracts necessary arguments. Updates the state with analysis results.
    """
    print("\n" + "="*50)
    print("🧠 Analyzing user intent...")
    print("="*50)

    user_input = state['user_input']
    print(f"\n📝 User Input: \"{user_input}\"")

    # Store current collected_info to track changes later
    state['previous_collected_info'] = state.get('collected_info', {}).copy()

    # Build comprehensive context for the LLM
    context = context_manager.build_enhanced_context(state)
    history_context = context.get('conversation', 'No history.')

    print("\n📜 Context Summary:")
    print(f"- Active Event ID: {state.get('active_event_id', 'None')}")
    print(f"- Last Tool Used: {state.get('last_successful_tool', 'None')}")
    print(f"- Conversation History Length: {len(state.get('conversation_history', []))} messages")
    print(f"- Previously Collected Info: {json.dumps(state.get('collected_info', {}), indent=2)}")

    prompt = f"""You are Mogambo, an intelligent calendar assistant created by Waseem M Ansari at WSMAISYS lab.

Current Date/Time: {current_time_str()}
Conversation History:
{history_context}

Previously Collected Information:
{json.dumps(state.get('collected_info', {}), indent=2)}

Active Event Context:
{json.dumps(context.get('active_event', {}), indent=2)}

User Input: "{user_input}"

Your task: Analyze the user's intent and determine the appropriate action.

Calendar Operations Available:
- create_calendar_event: Needs 'summary', 'start_time', 'end_time'. Optional: 'description', 'location'.
- retrieve_calendar_events: Needs 'time_min', 'time_max'. Optional: 'summary', 'location'.
- update_calendar_event: Needs 'event_id' (or context to infer it) and at least one field to update (e.g., 'summary', 'start_time', 'end_time', 'description', 'location').
- delete_calendar_event: Needs 'event_id' (or context to infer it).
- find_freebusy: Needs 'time_min', 'time_max'.

Special Handling for Event IDs:
1. If user references a previous event (e.g., "that meeting", "the event"), try to use 'active_event_id' from context.
2. If user selects by number (e.g., "the first one"), preserve the numeric reference for later resolution in the tool node.
3. Explicit event IDs are 32-character alphanumeric strings.

Decision Logic:
1. If the user's request is a general question or statement not related to calendar operations (e.g., greetings, small talk) → use "general_response".
2. If a calendar operation is clearly intended and enough information is provided → use the appropriate tool.
3. If a calendar operation is intended but critical information is missing → use "need_clarification" and specify 'missing_fields' and 'clarification_question'.
4. If the user's input seems to continue a previous operation (e.g., providing missing details) → merge new info with 'collected_info'.

For date/time parsing, be flexible:
- "today" = current date
- "tomorrow" = next day
- "next week" = 7 days from now
- "this morning", "this afternoon", "this evening"
- "in 2 hours", "in 30 minutes"
- Convert all dates/times to ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS+05:30 for IST).

Consider:
- References to previous events or context from the conversation history.
- Ambiguous terms that need clarification.
- Implied modifications to existing events.
- Time-based context and references.

Respond with JSON matching the `IntentAnalysis` structure:
{{
  "tool": "tool_name",
  "complete_args": {{"key": "value"}},
  "missing_fields": ["field1", "field2"],
  "clarification_question": "Specific question if needed",
  "confidence": 0.95,
  "reasoning": "Brief explanation of decision"
}}"""

    messages = [
        {"role": "system", "content": "You are an expert intent classifier. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ]

    analysis: IntentAnalysis = get_structured_response(llm_for_intent_and_tool, messages, IntentAnalysis, state)

    # Merge new arguments with existing collected_info
    current_collected = state.get('collected_info', {})
    new_collected = analysis.complete_args
    merged_collected = {**current_collected, **new_collected}

    # Update state with analysis results
    state['analysis_result'] = analysis
    state['intended_tool'] = analysis.tool.value
    state['collected_info'] = merged_collected
    state['missing_required_fields'] = analysis.missing_fields
    state['clarification_question'] = analysis.clarification_question
    state['pending_clarification'] = analysis.tool == ToolType.NEED_CLARIFICATION
    state['is_general_conversation'] = analysis.tool == ToolType.GENERAL_RESPONSE

    # Set flow control flags
    if analysis.tool in [ToolType.GENERAL_RESPONSE, ToolType.NEED_CLARIFICATION]:
        state['skip_to_response'] = True

    # Print detailed analysis results for debugging
    print("\n" + "="*50)
    print("🎯 Intent Analysis Results:")
    print("="*50)
    print(f"\n🔍 Detected Intent: {analysis.tool.value}")
    print(f"📊 Confidence: {analysis.confidence:.2f}")
    print(f"💭 Reasoning: {analysis.reasoning}")

    if merged_collected:
        print("\n📝 Collected Information:")
        print(json.dumps(merged_collected, indent=2))

    if analysis.missing_fields:
        print("\n❓ Missing Required Fields:")
        for field in analysis.missing_fields:
            print(f"- {field}")

    if analysis.clarification_question:
        print(f"\n❔ Clarification Needed: {analysis.clarification_question}")

    print("\n" + "="*50)

    return state

# --- Calendar Tool Nodes ---
def _resolve_event_reference(state: AgentState) -> Optional[str]:
    """
    Smart event resolution with context awareness and fuzzy matching.
    Returns event_id if found, None if needs clarification or no match.
    Updates state['event_context'] and state['last_event_matches'].
    """
    service = state.get('service')
    if not service:
        print("ERROR: Calendar service not available for event resolution.")
        return None

    event_id: Optional[str] = None
    calendar_id = state.get('collected_info', {}).get('calendar_id', 'primary')
    user_input = state['user_input'].lower()
    
    # 1. Try active event first if user refers to "this event" or it's the last active one
    if state.get('active_event_id'):
        try:
            event = service.events().get(calendarId=calendar_id, eventId=state['active_event_id']).execute()
            if any(ref in user_input for ref in ['this event', 'that event', 'current event']) or \
               fuzz.ratio(event.get('summary', '').lower(), user_input) > 70: # High fuzzy match
                state['event_context'] = event
                return state['active_event_id']
        except Exception:
            print(f"DEBUG: Active event ID {state['active_event_id']} is invalid or not found. Clearing.")
            state['active_event_id'] = '' # Clear invalid active event

    # 2. Try to resolve from user input with temporal/numeric references
    temporal_numeric_refs = {
        'last': -1, 'previous': -1, 'next': 0, 'upcoming': 0, # 'next' and 'upcoming' usually refer to the first in a list
        'first': 0, 'second': 1, 'third': 2, '1st': 0, '2nd': 1, '3rd': 2
    }
    
    if state.get('last_event_matches'):
        for ref, index in temporal_numeric_refs.items():
            if ref in user_input:
                try:
                    # Handle "next" and "upcoming" to refer to the first event in the list
                    if ref in ['next', 'upcoming'] and len(state['last_event_matches']) > 0:
                        state['event_context'] = state['last_event_matches'][0]
                        return state['last_event_matches'][0]['id']
                    elif 0 <= index < len(state['last_event_matches']):
                        state['event_context'] = state['last_event_matches'][index]
                        return state['last_event_matches'][index]['id']
                except IndexError:
                    print(f"DEBUG: Index {index} out of bounds for last_event_matches.")
                    pass
    
    # 3. Try fuzzy matching with recent events from conversation history
    recent_matches_from_history: List[Dict[str, Any]] = []
    for msg in reversed(state.get('conversation_history', [])):
        if msg.get('role') == 'assistant' and 'event' in msg.get('content', '').lower():
            ids = re.findall(r'ID: ([a-zA-Z0-9_-]+)', msg.get('content', ''))
            for eid in ids:
                try:
                    event = service.events().get(calendarId=calendar_id, eventId=eid).execute()
                    recent_matches_from_history.append({
                        'id': eid,
                        'summary': event.get('summary', ''),
                        'start': event['start'].get('dateTime', event['start'].get('date')),
                        'source': 'history'
                    })
                except Exception:
                    continue
    
    combined_recent_matches = list(state.get('last_event_matches', [])) + recent_matches_from_history
    
    if combined_recent_matches:
        best_match_id = None
        best_score = 0.0
        for event in combined_recent_matches:
            event_summary = event.get('summary', '').lower()
            summary_score = fuzz.ratio(user_input, event_summary)
            
            # Temporal relevance boost
            time_relevance_boost = 1.0
            if 'start' in event:
                try:
                    start_val = event['start']
                    if isinstance(start_val, dict):
                        start_str = start_val.get('dateTime') or start_val.get('date')
                    else:
                        start_str = start_val
                    if isinstance(start_str, str):
                        event_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                        if event_time.tzinfo is None:
                            event_time = event_time.replace(tzinfo=pytz.UTC)
                        else:
                            event_time = event_time.astimezone(pytz.UTC)
                        now_utc = datetime.now(pytz.UTC)
                        time_diff_hours = abs((event_time - now_utc).total_seconds() / 3600)
                        if time_diff_hours < 24: time_relevance_boost = 1.2
                        elif time_diff_hours < 168: time_relevance_boost = 1.1
                except (ValueError, TypeError, AttributeError): pass
            
            # Source boost (history matches are often more relevant)
            source_boost = 1.2 if event.get('source') == 'history' else 1.0
            
            final_score = summary_score * time_relevance_boost * source_boost
            
            if final_score > best_score and final_score > 75: # High confidence threshold
                best_score = final_score
                best_match_id = event['id']
        
        if best_match_id:
            try:
                event = service.events().get(calendarId=calendar_id, eventId=best_match_id).execute()
                state['event_context'] = event
                return best_match_id
            except Exception:
                print(f"DEBUG: Best fuzzy match event ID {best_match_id} not found or invalid.")
                pass

    # 4. Search by extended context (summary, date, description from collected_info)
    # This helper function needs to be defined or mocked if not already present
    def find_events_by_context(service_obj, cal_id, context_data):
        """
        Searches for events in the calendar matching the provided context_data (summary, time_min, time_max).
        Returns a list of matching events.
        """
        print(f"DEBUG: Searching events by context: {context_data}")
        summary = context_data.get('summary', '').lower()
        time_min = context_data.get('time_min')
        time_max = context_data.get('time_max')
        try:
            events_result = service_obj.events().list(
                calendarId=cal_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            matches = []
            for event in events:
                event_summary = event.get('summary', '').lower()
                if summary and summary in event_summary:
                    matches.append({
                        'id': event.get('id'),
                        'summary': event.get('summary', ''),
                        'start': event.get('start', {})
                    })
            print(f"DEBUG: Found {len(matches)} matching events.")
            return matches
        except Exception as e:
            print(f"ERROR: Failed to search events by context: {e}")
            return []

    context_info = state.get('collected_info', {})
    if context_info.get('summary') or context_info.get('date'):
        search_matches = find_events_by_context(service, calendar_id, context_info)
        if search_matches:
            state['last_event_matches'] = search_matches # Store all matches
            if len(search_matches) == 1:
                state['event_context'] = search_matches[0]
                return search_matches[0]['id']
            else:
                # Multiple matches, need clarification
                clarification_details = []
                for i, match in enumerate(search_matches[:3], 1): # Show top 3
                    start_time = match.get('start', 'unknown time')
                    summary = match.get('summary', 'Untitled event')
                    clarification_details.append(f"{i}. '{summary}' at {start_time}")
                
                state['clarification_context'] = {
                    'type': 'event_selection',
                    'reason': 'Multiple matching events found',
                    'matches': search_matches,
                    'details': clarification_details,
                    'suggestion': "Please specify which event you mean by its number or more details."
                }
                print(f"DEBUG: Multiple events found, requiring clarification: {clarification_details}")
                return None # Indicate need for clarification

    # No event found
    state['clarification_context'] = {
        'type': 'no_matches',
        'reason': 'No matching events found based on current context.',
        'searched_context': context_info,
        'suggestion': "Try providing more specific details about the event you're looking for (e.g., exact title, date)."
    }
    print("DEBUG: No event reference resolved.")
    return None

def create_calendar_event_node(state: AgentState) -> AgentState:
    """Creates a new calendar event using collected information."""
    print("📅 Entering create_calendar_event_node...")
    service = state.get('service')
    if not service:
        state['tool_output'] = "Calendar service is not available. Please ensure you are logged in."
        state['error_message'] = "Calendar service unavailable."
        return state

    args = state['collected_info']
    required_fields = ['summary', 'start_time', 'end_time']
    missing = [field for field in required_fields if not args.get(field)]

    if missing:
        state['tool_output'] = f"Missing required fields to create event: {', '.join(missing)}."
        state['error_message'] = "Missing required fields."
        state['missing_required_fields'] = missing
        state['pending_clarification'] = True
        return state

    try:
        start_time_iso = parse_datetime_flexible(args['start_time'])
        end_time_iso = parse_datetime_flexible(args['end_time'])

        event = {
            'summary': args['summary'],
            'start': {'dateTime': start_time_iso, 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': end_time_iso, 'timeZone': 'Asia/Kolkata'},
        }
        if args.get('description'): event['description'] = args['description']
        if args.get('location'): event['location'] = args['location']

        calendar_id = args.get('calendar_id', 'primary')
        created_event = service.events().insert(calendarId=calendar_id, body=event).execute()

        event_id = created_event.get('id')
        if event_id:
            state['active_event_id'] = event_id
            state['last_successful_tool'] = ToolType.CREATE_EVENT.value
            state['tool_output'] = (
                f"✅ Event '{args['summary']}' created successfully!\n"
                f"📅 Start: {start_time_iso}\n"
                f"⏰ End: {end_time_iso}\n"
                f"🔗 Event ID: {event_id}"
            )
            print(f"DEBUG: Event created: {state['tool_output']}")
        else:
            state['tool_output'] = "Failed to retrieve event ID after creation."
            state['error_message'] = "Event creation successful but ID not returned."
            print("ERROR: Event created but ID not returned.")

    except Exception as e:
        state['error_message'] = f"Failed to create event: {str(e)}"
        state['tool_output'] = f"❌ Failed to create event: {str(e)}"
        print(f"ERROR: Event creation failed: {e}")

    return state

def retrieve_calendar_events_node(state: AgentState) -> AgentState:
    """Retrieves calendar events based on time range and other filters."""
    print("📋 Entering retrieve_calendar_events_node...")
    service = state.get('service')
    if not service:
        state['tool_output'] = "Calendar service is not available. Please ensure you are logged in."
        state['error_message'] = "Calendar service unavailable."
        return state

    args = state['collected_info']
    if not args.get('time_min') or not args.get('time_max'):
        state['tool_output'] = "Missing time range for event retrieval. Please specify a start and end time."
        state['error_message'] = "Missing time range."
        state['missing_required_fields'] = ['time_min', 'time_max']
        state['pending_clarification'] = True
        return state

    try:
        tz = pytz.timezone('Asia/Kolkata')
        # Ensure full day coverage for date-only inputs
        time_min_dt = tz.localize(datetime.combine(datetime.strptime(args['time_min'][:10], "%Y-%m-%d").date(), time(0, 0, 0)))
        time_max_dt = tz.localize(datetime.combine(datetime.strptime(args['time_max'][:10], "%Y-%m-%d").date(), time(23, 59, 59)))

        time_min_iso = time_min_dt.isoformat()
        time_max_iso = time_max_dt.isoformat()

        calendar_id = args.get('calendar_id', 'primary')
        cache_key = f"events_{calendar_id}_{time_min_iso}_{time_max_iso}"
        cached_events = GlobalCache.instance().get(cache_key)

        if cached_events:
            events = cached_events
            print("DEBUG: Using cached events for retrieval.")
        else:
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min_iso,
                timeMax=time_max_iso,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            GlobalCache.instance().set(cache_key, events)
            print("DEBUG: Fetched events from calendar API.")

        if not events:
            state['tool_output'] = f"I've checked your calendar and it appears to be clear between {time_min_dt.strftime('%I:%M %p on %b %d')} and {time_max_dt.strftime('%I:%M %p on %b %d')}. You have no scheduled events during this time period."
        else:
            event_list_str = []
            for i, event in enumerate(events, 1):
                start = event['start'].get('dateTime', event['start'].get('date'))
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00')).astimezone(tz)
                    formatted_start = start_dt.strftime("%I:%M %p on %b %d")
                else:
                    formatted_start = f"All day on {start}"

                summary = event.get('summary', 'Untitled Event')
                location = f" at {event['location']}" if event.get('location') else ""
                event_id = event.get('id', '')
                event_list_str.append(f"{i}. {summary} - {formatted_start}{location} (ID: {event_id[:8]}...)")

            if len(events) == 1:
                state['active_event_id'] = events[0].get('id', '')
                state['event_context'] = events[0]
                print(f"DEBUG: Set active event ID: {state['active_event_id']}")

            state['last_event_matches'] = events # Store for potential future reference
            state['last_successful_tool'] = ToolType.RETRIEVE_EVENTS.value
            state['tool_output'] = (
                f"Here's what I found on your calendar:\n\n" +
                "\n".join(event_list_str) +
                f"\n\nTotal: {len(events)} event{'s' if len(events) != 1 else ''}."
            )
            print(f"DEBUG: Successfully processed {len(events)} events.")

    except Exception as e:
        state['error_message'] = f"Failed to retrieve events: {str(e)}"
        state['tool_output'] = f"❌ Failed to retrieve events: {str(e)}"
        print(f"ERROR: Event retrieval failed: {e}")

    return state

def update_calendar_event_node(state: AgentState) -> AgentState:
    """Updates an existing calendar event."""
    print("✏️ Entering update_calendar_event_node...")
    service = state.get('service')
    if not service:
        state['tool_output'] = "Calendar service is not available. Please ensure you are logged in."
        state['error_message'] = "Calendar service unavailable."
        return state

    args = state['collected_info']
    calendar_id = args.get('calendar_id', 'primary')

    event_id = args.get('event_id') or _resolve_event_reference(state)

    if not event_id:
        state['tool_output'] = "I couldn't identify which event you want to update. Please be more specific."
        state['error_message'] = "Event ID not resolved for update."
        state['missing_required_fields'] = ['event_id']
        state['pending_clarification'] = True
        return state

    try:
        existing_event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        updates_made = []

        if args.get('summary') and existing_event.get('summary') != args['summary']:
            existing_event['summary'] = args['summary']
            updates_made.append(f"Summary to '{args['summary']}'")
        if args.get('description') and existing_event.get('description') != args['description']:
            existing_event['description'] = args['description']
            updates_made.append(f"Description to '{args['description']}'")
        if args.get('location') and existing_event.get('location') != args['location']:
            existing_event['location'] = args['location']
            updates_made.append(f"Location to '{args['location']}'")

        # Handle time updates
        if args.get('start_time'):
            new_start_time_iso = parse_datetime_flexible(args['start_time'])
            if existing_event['start'].get('dateTime') != new_start_time_iso:
                existing_event['start']['dateTime'] = new_start_time_iso
                updates_made.append(f"Start time to {new_start_time_iso}")
        if args.get('end_time'):
            new_end_time_iso = parse_datetime_flexible(args['end_time'])
            if existing_event['end'].get('dateTime') != new_end_time_iso:
                existing_event['end']['dateTime'] = new_end_time_iso
                updates_made.append(f"End time to {new_end_time_iso}")

        if updates_made:
            updated_event = service.events().update(calendarId=calendar_id, eventId=event_id, body=existing_event).execute()
            state['active_event_id'] = event_id
            state['last_successful_tool'] = ToolType.UPDATE_EVENT.value
            state['tool_output'] = (
                f"✅ Event '{updated_event.get('summary', 'Untitled Event')}' updated successfully!\n"
                f"🔗 Event ID: {event_id}\n"
                f"Updated: {', '.join(updates_made)}"
            )
            print(f"DEBUG: Event updated: {state['tool_output']}")
        else:
            state['tool_output'] = "No changes detected to update the event. Please specify what you'd like to change."
            print("DEBUG: No changes requested for event update.")

    except Exception as e:
        state['error_message'] = f"Failed to update event: {str(e)}"
        state['tool_output'] = f"❌ Failed to update event: {str(e)}"
        print(f"ERROR: Event update failed: {e}")

    return state

def delete_calendar_event_node(state: AgentState) -> AgentState:
    """Deletes a calendar event."""
    print("🗑️ Entering delete_calendar_event_node...")
    service = state.get('service')
    if not service:
        state['tool_output'] = "Calendar service is not available. Please ensure you are logged in."
        state['error_message'] = "Calendar service unavailable."
        return state

    args = state['collected_info']
    print(f"DEBUG: delete_calendar_event_node args type: {type(args)}, value: {args}")
    if not isinstance(args, dict):
        try:
            args = dict(args)
        except Exception as e:
            state['error_message'] = f"Invalid collected_info type: {type(args)}. Error: {e}"
            state['tool_output'] = state['error_message']
            return state
    calendar_id = args.get('calendar_id', 'primary')

    event_id = args.get('event_id') or _resolve_event_reference(state)

    if not event_id:
        state['tool_output'] = "I couldn't identify which event you want to delete. Please be more specific."
        state['error_message'] = "Event ID not resolved for deletion."
        state['missing_required_fields'] = ['event_id']
        state['pending_clarification'] = True
        return state

    try:
        # Get event details before deleting for a better confirmation message
        event_details = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        event_summary = event_details.get('summary', 'Unknown event')
        event_start_time = event_details['start'].get('dateTime', event_details['start'].get('date'))

        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()

        state['last_successful_tool'] = ToolType.DELETE_EVENT.value
        state['active_event_id'] = "" # Clear active event after deletion
        state['tool_output'] = (
            f"✅ Successfully deleted event '{event_summary}' "
            f"scheduled for {event_start_time}.\n"
            f"🔗 Event ID: {event_id}"
        )
        print(f"DEBUG: Event deleted: {state['tool_output']}")

    except Exception as e:
        state['error_message'] = f"Failed to delete event: {str(e)}"
        state['tool_output'] = f"❌ Failed to delete event: {str(e)}"
        print(f"ERROR: Event deletion failed: {e}")

    return state

def find_freebusy_node(state: AgentState) -> AgentState:
    """Checks calendar availability for a given time range."""
    print("🔍 Entering find_freebusy_node...")
    service = state.get('service')
    if not service:
        state['tool_output'] = "Calendar service is not available. Please ensure you are logged in."
        state['error_message'] = "Calendar service unavailable."
        return state

    args = state['collected_info']
    if not args.get('time_min') or not args.get('time_max'):
        state['tool_output'] = "Missing time range for availability check. Please specify a start and end time."
        state['error_message'] = "Missing time range."
        state['missing_required_fields'] = ['time_min', 'time_max']
        state['pending_clarification'] = True
        return state

    try:
        time_min_iso = parse_datetime_flexible(args['time_min'])
        time_max_iso = parse_datetime_flexible(args['time_max'])
        calendar_id = args.get('calendar_id', 'primary')

        body = {
            "timeMin": time_min_iso,
            "timeMax": time_max_iso,
            "items": [{"id": calendar_id}]
        }

        response = service.freebusy().query(body=body).execute()
        busy_times = response['calendars'][calendar_id]['busy']

        if not busy_times:
            state['tool_output'] = f"✅ You are completely free from {time_min_iso} to {time_max_iso}."
        else:
            busy_list = []
            for busy_time in busy_times:
                busy_list.append(f"• {busy_time['start']} to {busy_time['end']}")
            state['tool_output'] = (
                f"⏰ Busy periods found between {time_min_iso} and {time_max_iso}:\n" +
                "\n".join(busy_list)
            )
        state['last_successful_tool'] = ToolType.FIND_FREEBUSY.value
        print(f"DEBUG: Free/busy check result: {state['tool_output']}")

    except Exception as e:
        state['error_message'] = f"Failed to check availability: {str(e)}"
        state['tool_output'] = f"❌ Failed to check availability: {str(e)}"
        print(f"ERROR: Free/busy check failed: {e}")

    return state

def generate_intelligent_response_node(state: AgentState) -> AgentState:
    """
    Generates the final conversational response to the user,
    incorporating tool outputs, errors, and context.
    """
    print("\n" + "="*50)
    print("🗣️ Generating intelligent response...")
    print("="*50)

    user_input = state['user_input']
    tool_output = state.get('tool_output', '')
    error_msg = state.get('error_message', '')
    clarification_q = state.get('clarification_question', '')
    pending_clarification = state.get('pending_clarification', False)
    is_general_conversation = state.get('is_general_conversation', False)

    # Check response cache first (only for non-error states)
    cache_key = f"response_{user_input}_{state.get('intended_tool', '')}_{tool_output}_{error_msg}"
    cached_response = GlobalCache.instance().get(cache_key)
    if cached_response and not error_msg:
        state['final_response_text'] = cached_response
        print("DEBUG: Using cached response.")
        return state

    context = context_manager.build_enhanced_context(state)
    history_context = context.get('conversation', 'No history.')

    # Determine the core message based on flow
    core_message = ""
    if error_msg:
        core_message = f"I encountered an issue: {error_msg}. "
        if pending_clarification:
            core_message += clarification_q
        else:
            core_message += "Please try again or rephrase your request."
    elif pending_clarification:
        core_message = clarification_q
        if state.get('clarification_context'):
            ctx = state['clarification_context']
            if ctx.get('type') == 'event_selection' and ctx.get('details'):
                core_message += "\n" + "Possible matches:\n" + "\n".join(ctx['details'])
                core_message += f"\n{ctx.get('suggestion', '')}"
            elif ctx.get('type') == 'no_matches':
                core_message += f"\n{ctx.get('suggestion', '')}"
    elif is_general_conversation:
        # If it's a general conversation, the LLM will generate the full response
        pass
    elif tool_output:
        core_message = tool_output
    else:
        core_message = "I'm not sure how to proceed. Can you please clarify?"

    # Construct the prompt for the LLM to generate a natural response
    prompt = f"""You are Mogambo, a helpful and friendly calendar assistant. Your goal is to provide clear, concise, and conversational responses.

Current Date/Time: {current_time_str()}
User's original input: "{user_input}"
Conversation History:
{history_context}

Internal State and Tool Execution Result:
- Intended Tool: {state.get('intended_tool', 'N/A')}
- Tool Output: {tool_output if tool_output else 'N/A'}
- Error Message: {error_msg if error_msg else 'N/A'}
- Clarification Needed: {pending_clarification}
- Clarification Question: {clarification_q if clarification_q else 'N/A'}
- Is General Conversation: {is_general_conversation}
- Active Event ID: {state.get('active_event_id', 'None')}
- Last Successful Tool: {state.get('last_successful_tool', 'None')}

Based on the above information, generate a natural, conversational response to the user.
If there's a tool output, summarize it clearly. If clarification is needed, ask the question directly.
If it's a general conversation, respond appropriately. If there's an error, explain it simply and suggest next steps.
Ensure your response is helpful and guides the user.
"""
    messages = [
        {"role": "system", "content": "You are Mogambo, a helpful calendar assistant. Respond in a natural, conversational way."},
        {"role": "user", "content": prompt}
    ]

    response = llm_for_response.invoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Add personality touch and dynamic suggestions
    final_response = _add_personality_touch(user_input, response_text, state)

    state['final_response_text'] = final_response
    GlobalCache.instance().set(cache_key, final_response) # Cache the generated response

    print("\n" + "="*50)
    print("🗣️ Final Generated Response:")
    print("="*50)
    print(final_response)
    print("="*50)

    return state

def text_to_speech_node(state: AgentState) -> AgentState:
    """
    Converts the final response text to speech using gTTS and plays the audio.
    Handles cleanup and platform-specific playback.
    """
    audio_file = "response_tts.mp3"
    try:
        text = state.get('final_response_text', '').strip()
        if not text:
            state['error_message'] = "No text to convert to speech."
            print("ERROR: No text for TTS conversion.")
            return state

        # Clean up existing audio file if it exists
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                print(f"DEBUG: Removed old audio file: {audio_file}")
            except OSError as e:
                print(f"WARNING: Could not remove old audio file {audio_file}: {e}")

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_file)

        if os.path.exists(audio_file):
            _play_audio_platform_agnostic(audio_file)
            state['audio_file'] = audio_file
            print(f"DEBUG: Audio saved and played: {audio_file}")
        else:
            state['error_message'] = "Failed to create audio file for TTS."
            print("ERROR: Failed to create audio file for TTS.")

    except Exception as e:
        state['error_message'] = f"TTS error: {str(e)}"
        print(f"ERROR: TTS processing error: {e}")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except OSError:
                pass # Ignore if cleanup fails again

    return state

# --- Dynamic Clarification and Personality ---
def _handle_dynamic_clarification(state: AgentState) -> Dict[str, Any]:
    """
    Dynamically generates context-aware clarification requests with intelligent suggestions.
    This function is intended to be called by the response generation node.
    """
    collected = state.get('collected_info', {})
    missing = state.get('missing_required_fields', [])
    context = context_manager.build_enhanced_context(state)
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)

    clarification = {
        'type': 'general', 'questions': [], 'suggestions': [],
        'context_hints': [], 'quick_fixes': [], 'examples': []
    }

    if missing:
        for field in missing:
            if field == 'summary':
                clarification['questions'].append("What would you like to name this event?")
                if context.get('active_event'):
                    clarification['suggestions'].append(f"Perhaps something similar to '{context['active_event'].get('summary', '')}'?")
            elif field in ['start_time', 'end_time']:
                is_start = field == 'start_time'
                clarification['questions'].append(f"When would you like the event to {'start' if is_start else 'end'}?")
                if is_start:
                    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    clarification['quick_fixes'].append(f"Start at {next_hour.strftime('%I:%M %p')}")
                elif collected.get('start_time'):
                    try:
                        start_dt = datetime.fromisoformat(collected['start_time']).astimezone(tz)
                        suggested_end = start_dt + timedelta(hours=1)
                        clarification['quick_fixes'].append(f"End at {suggested_end.strftime('%I:%M %p')}")
                    except ValueError: pass
                clarification['examples'].extend(["'tomorrow at 3pm'", "'next Monday morning'", "'in 2 hours'"])
            elif field == 'description':
                clarification['questions'].append("Would you like to add any details or agenda for this event?")
            elif field == 'location':
                clarification['questions'].append("Where will this event take place?")
                recent_locations = {msg['content'] for msg in state.get('conversation_history', []) if 'location' in msg.get('content', '').lower()}
                if recent_locations:
                    clarification['suggestions'].append("Recent locations used: " + ", ".join(list(recent_locations)[:2]))

    if context.get('active_event'):
        event = context['active_event']
        clarification['context_hints'].append(f"We're currently working with the event '{event.get('summary', '')}' scheduled for {event.get('start', {}).get('dateTime', 'unknown time')}.")
        if state.get('intended_tool') == 'update_calendar_event':
            clarification['context_hints'].append("You can modify its time, description, title, or location.")

    last_tool = state.get('last_successful_tool', '')
    if last_tool == 'create_calendar_event':
        clarification['suggestions'].extend(["You can add a description or location.", "You can set a reminder."])
    elif last_tool == 'update_calendar_event':
        clarification['suggestions'].extend(["You can update any detail of the event.", "Say 'move it to tomorrow' to reschedule."])
    elif last_tool == 'delete_calendar_event':
        clarification['suggestions'].extend(["You can confirm by saying 'yes, delete it'.", "Or cancel by saying 'no, keep it'."])

    if state.get('error_message'):
        clarification['quick_fixes'].extend(["Try rephrasing your request.", "Use specific dates and times."])

    preferences = context.get('preferences', {})
    if preferences:
        if 'preferred_time' in preferences:
            clarification['suggestions'].append(f"I notice you often schedule events in the {preferences['preferred_time']}. Would you like to schedule this then?")
        if 'default_duration' in preferences:
            clarification['suggestions'].append(f"Your meetings typically last {preferences['default_duration']} minutes. Shall we use that duration?")

    return clarification

def _add_personality_touch(user_input: str, response_text: str, state: AgentState) -> str:
    """Adds a personality touch and dynamic suggestions to the response."""
    clarification_info = _handle_dynamic_clarification(state)

    if clarification_info['questions'] and state.get('pending_clarification'):
        response_text += "\n\n" + "\n".join(clarification_info['questions'])
    if clarification_info['suggestions']:
        response_text += "\n\nHelpful hint: " + clarification_info['suggestions'][0]
    if clarification_info['context_hints']:
        response_text = clarification_info['context_hints'][0] + "\n" + response_text

    if "created by Waseem M Ansari" not in response_text and "introduce" in user_input.lower():
        return f"Hello! I'm Mogambo, your calendar assistant created by Waseem M Ansari at WSMAISYS lab. {response_text}"
    return response_text

# --- LangGraph Workflow Definition ---
def create_intelligent_workflow() -> StateGraph:
    """
    Creates and compiles the LangGraph workflow for the calendar assistant.
    Defines nodes and conditional edges for state transitions.
    """
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node('detect_mode', detect_mode)
    graph.add_node('voice_input', voice_detection_node)
    graph.add_node('intent_analysis', intelligent_intent_analysis_node)
    graph.add_node('create_event', create_calendar_event_node)
    graph.add_node('retrieve_events', retrieve_calendar_events_node)
    graph.add_node('update_event', update_calendar_event_node)
    graph.add_node('delete_event', delete_calendar_event_node)
    graph.add_node('find_freebusy', find_freebusy_node)
    graph.add_node('generate_response', generate_intelligent_response_node)
    graph.add_node('text_to_speech', text_to_speech_node)

    # Define graph entry point
    graph.add_edge(START, 'detect_mode')

    # Conditional routing based on input mode (voice or text)
    graph.add_conditional_edges(
        'detect_mode',
        lambda state: 'voice_input' if state.get('is_voice_input') else 'intent_analysis',
        {'voice_input': 'voice_input', 'intent_analysis': 'intent_analysis'}
    )

    # After voice input, proceed to intent analysis
    graph.add_edge('voice_input', 'intent_analysis')

    # Conditional routing after intent analysis to specific tool nodes or response generation
    def route_after_intent(state: AgentState) -> str:
        if state.get('skip_to_response'):
            return 'generate_response'

        tool_mapping = {
            ToolType.CREATE_EVENT.value: 'create_event',
            ToolType.RETRIEVE_EVENTS.value: 'retrieve_events',
            ToolType.UPDATE_EVENT.value: 'update_event',
            ToolType.DELETE_EVENT.value: 'delete_event',
            ToolType.FIND_FREEBUSY.value: 'find_freebusy',
            ToolType.NEED_CLARIFICATION.value: 'generate_response', # Clarification handled by response node
            ToolType.GENERAL_RESPONSE.value: 'generate_response', # General conversation handled by response node
            ToolType.NONE.value: 'generate_response' # Fallback
        }
        return tool_mapping.get(state.get('intended_tool', ToolType.NONE.value), 'generate_response')

    graph.add_conditional_edges(
        'intent_analysis',
        route_after_intent,
        {
            'create_event': 'create_event',
            'retrieve_events': 'retrieve_events',
            'update_event': 'update_event',
            'delete_event': 'delete_event',
            'find_freebusy': 'find_freebusy',
            'generate_response': 'generate_response' # Direct path for clarification/general response
        }
    )

    # All tool nodes transition to response generation
    for tool_node in ['create_event', 'retrieve_events', 'update_event', 'delete_event', 'find_freebusy']:
        graph.add_edge(tool_node, 'generate_response')

    # Final routing from response generation to text-to-speech or END
    graph.add_conditional_edges(
        'generate_response',
        lambda state: 'text_to_speech' if state.get('is_voice_input') else END,
        {'text_to_speech': 'text_to_speech', END: END}
    )

    # Text-to-speech node always leads to END
    graph.add_edge('text_to_speech', END)

    return graph.compile()

# Create the workflow instance
workflow = create_intelligent_workflow()

# --- State Initialization Helper ---
def create_initial_state(
    user_input: str,
    session_id: str,
    service: Any,
    is_voice: bool = False,
    previous_state: Optional[AgentState] = None
) -> AgentState:
    """
    Factory method for creating an initial AgentState.
    Allows for carrying over context from a previous state for multi-turn conversations.
    """
    if previous_state:
        # Preserve context from previous interaction
        state: AgentState = {
            'user_input': user_input,
            'is_voice_input': is_voice,
            'session_id': session_id,
            'service': service or previous_state.get('service'), # Use new service if provided, else old
            'final_response_text': "",
            'error_message': "",
            'conversation_history': previous_state.get('conversation_history', []),
            'collected_info': previous_state.get('collected_info', {}),
            'intended_tool': "",
            'active_event_id': previous_state.get('active_event_id', ''),
            'last_successful_tool': previous_state.get('last_successful_tool', ''),
            'is_general_conversation': False,
            'analysis_result': None,
            'context_stack': previous_state.get('context_stack', []),
            'tool_output': "",
            'skip_to_response': False,
            'pending_clarification': False,
            'missing_required_fields': [],
            'clarification_question': "",
            'last_event_matches': previous_state.get('last_event_matches', []),
            'event_context': previous_state.get('event_context', {}),
            'clarification_context': previous_state.get('clarification_context', {}),
            'user_preferences': previous_state.get('user_preferences', {}),
            'audio_file': "",
            'previous_collected_info': previous_state.get('collected_info', {}) # Snapshot for change tracking
        }
    else:
        # Brand new initial state
        state: AgentState = {
            'user_input': user_input,
            'is_voice_input': is_voice,
            'session_id': session_id,
            'service': service,
            'final_response_text': "",
            'error_message': "",
            'conversation_history': [],
            'collected_info': {},
            'intended_tool': "",
            'active_event_id': "",
            'last_successful_tool': "",
            'is_general_conversation': False,
            'analysis_result': None,
            'context_stack': [],
            'tool_output': "",
            'skip_to_response': False,
            'pending_clarification': False,
            'missing_required_fields': [],
            'clarification_question': "",
            'last_event_matches': [],
            'event_context': {},
            'clarification_context': {},
            'user_preferences': {},
            'audio_file': "",
            'previous_collected_info': {}
        }

    # Add current user input to conversation history
    # This is handled in app.py before invoking the workflow to ensure it's always added.
    # If this function were called directly, it would add it here.
    # For the current app.py structure, this append might be redundant if app.py already does it.
    # Keeping it here for robustness if create_initial_state is used elsewhere.
    if not state['conversation_history'] or state['conversation_history'][-1].get('content') != user_input:
        state['conversation_history'].append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

    return state

# # Example of how to run the workflow (for demonstration, not part of the refactored agent.py)
# if __name__ == "__main__":
#     # This block demonstrates how the refactored code would be used.
#     # In a real application, 'service' would be an authenticated Google Calendar API client.
#     # For testing, you might mock it or provide a dummy object.

#     class MockCalendarService:
#         """A mock Google Calendar service for testing purposes."""
#         def events(self):
#             class MockEvents:
#                 def list(self, **kwargs):
#                     print(f"MockCalendarService: Listing events with {kwargs}")
#                     # Return a dummy event list for testing retrieve_calendar_events_node
#                     if "timeMin" in kwargs and "timeMax" in kwargs:
#                         return type('obj', (object,), {'execute': lambda: {
#                             'items': [
#                                 {
#                                     'id': 'mock_event_1',
#                                     'summary': 'Mock Event Today',
#                                     'start': {'dateTime': '2023-11-20T09:00:00+05:30'},
#                                     'end': {'dateTime': '2023-11-20T10:00:00+05:30'}
#                                 },
#                                 {
#                                     'id': 'mock_event_2',
#                                     'summary': 'Another Mock Event',
#                                     'start': {'dateTime': '2023-11-20T14:00:00+05:30'},
#                                     'end': {'dateTime': '2023-11-20T15:00:00+05:30'}
#                                 }
#                             ]
#                         }})()
#                     return self

#                 def get(self, **kwargs):
#                     print(f"MockCalendarService: Getting event with {kwargs}")
#                     # Return a dummy event for testing _resolve_event_reference and update/delete
#                     event_id = kwargs.get('eventId')
#                     if event_id == 'dummy_event_id_123':
#                         return type('obj', (object,), {'execute': lambda: {
#                             'id': 'dummy_event_id_123',
#                             'summary': 'Project Sync',
#                             'start': {'dateTime': '2023-11-21T10:00:00+05:30'},
#                             'end': {'dateTime': '2023-11-21T11:00:00+05:30'}
#                         }})()
#                     elif event_id == 'new_event_id_456':
#                          return type('obj', (object,), {'execute': lambda: {
#                             'id': 'new_event_id_456',
#                             'summary': 'New Meeting',
#                             'start': {'dateTime': '2023-11-22T09:00:00+05:30'},
#                             'end': {'dateTime': '2023-11-22T10:00:00+05:30'}
#                         }})()
#                     elif event_id == 'mock_event_1':
#                          return type('obj', (object,), {'execute': lambda: {
#                             'id': 'mock_event_1',
#                             'summary': 'Mock Event Today',
#                             'start': {'dateTime': '2023-11-20T09:00:00+05:30'},
#                             'end': {'dateTime': '2023-11-20T10:00:00+05:30'}
#                         }})()
#                     return type('obj', (object,), {'execute': lambda: {}})() # Default empty event

#                 def insert(self, **kwargs):
#                     print(f"MockCalendarService: Inserting event with {kwargs}")
#                     return type('obj', (object,), {'execute': lambda: {'id': 'new_event_id_456', 'summary': kwargs['body']['summary']}})()

#                 def update(self, **kwargs):
#                     print(f"MockCalendarService: Updating event with {kwargs}")
#                     # Simulate updating the summary and time
#                     updated_summary = kwargs['body'].get('summary', 'Updated Event')
#                     updated_start = kwargs['body'].get('start', {}).get('dateTime', 'N/A')
#                     return type('obj', (object,), {'execute': lambda: {'id': kwargs['eventId'], 'summary': updated_summary, 'start': {'dateTime': updated_start}}})()

#                 def delete(self, **kwargs):
#                     print(f"MockCalendarService: Deleting event with {kwargs}")
#                     return type('obj', (object,), {'execute': lambda: None})()

#                 def query(self, **kwargs):
#                     print(f"MockCalendarService: Querying freebusy with {kwargs}")
#                     # Always return free for mock
#                     return type('obj', (object,), {'execute': lambda: {'calendars': {'primary': {'busy': []}}}})()
#             return MockEvents()
#         def freebusy(self):
#             return self.events() # freebusy is also under events in this mock

#     mock_service = MockCalendarService()
#     session_id = "test_session_123"
#     current_state: Optional[AgentState] = None

#     print("\n--- Test Scenario 1: Create Event ---")
#     user_input_1 = "Create a meeting called 'Project Sync' tomorrow at 10 AM for 1 hour."
#     current_state = create_initial_state(user_input_1, session_id=session_id, service=mock_service)
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response: {final_state.get('final_response_text')}")
#     print(f"Active Event ID: {final_state.get('active_event_id')}")
#     print(f"Collected Info: {final_state.get('collected_info')}")
#     # Store the active event ID for subsequent tests
#     active_event_id_from_create = final_state.get('active_event_id')

#     print("\n--- Test Scenario 2: Retrieve Events ---")
#     user_input_2 = "What are my events for today?"
#     current_state = create_initial_state(user_input_2, session_id=session_id, service=mock_service, previous_state=final_state)
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response: {final_state.get('final_response_text')}")
#     print(f"Collected Info: {final_state.get('collected_info')}")

#     print("\n--- Test Scenario 3: General Conversation ---")
#     user_input_3 = "Hello Mogambo, how are you?"
#     current_state = create_initial_state(user_input_3, session_id=session_id, service=mock_service, previous_state=final_state)
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response: {final_state.get('final_response_text')}")

#     print("\n--- Test Scenario 4: Update Event (using active event from create) ---")
#     user_input_4 = "Update that meeting to start at 11 AM instead."
#     current_state = create_initial_state(user_input_4, session_id=session_id, service=mock_service, previous_state=final_state)
#     # Ensure the active_event_id is correctly passed for the update
#     current_state['active_event_id'] = active_event_id_from_create if active_event_id_from_create else 'dummy_event_id_123'
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response: {final_state.get('final_response_text')}")
#     print(f"Collected Info: {final_state.get('collected_info')}")

#     print("\n--- Test Scenario 5: Voice Input Simulation ---")
#     user_input_5 = "Simulate voice input for 'What is the weather like?'"
#     current_state = create_initial_state(user_input_5, session_id=session_id, service=mock_service, is_voice=True, previous_state=final_state)
#     # In a real scenario, voice_detection_node would transcribe. Here we just simulate the flag.
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response (Voice Simulated): {final_state.get('final_response_text')}")
#     print(f"Audio File Generated: {final_state.get('audio_file')}")

#     print("\n--- Test Scenario 6: Find Free/Busy ---")
#     user_input_6 = "Am I free tomorrow from 9 AM to 5 PM?"
#     current_state = create_initial_state(user_input_6, session_id=session_id, service=mock_service, previous_state=final_state)
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response: {final_state.get('final_response_text')}")
#     print(f"Collected Info: {final_state.get('collected_info')}")

#     print("\n--- Test Scenario 7: Delete Event (using a specific ID) ---")
#     user_input_7 = "Delete the event with ID mock_event_1."
#     current_state = create_initial_state(user_input_7, session_id=session_id, service=mock_service, previous_state=final_state)
#     final_state = workflow.invoke(current_state)
#     print(f"\nFinal Response: {final_state.get('final_response_text')}")
#     print(f"Collected Info: {final_state.get('collected_info')}")

#     # Clean up generated audio file if it exists
#     if os.path.exists("response_tts.mp3"):
#         os.remove("response_tts.mp3")
#         print("Cleaned up response_tts.mp3")
#     print("\n--- All test scenarios completed. ---")
# # Note: The above test scenarios are for demonstration purposes and would typically be run in a testing