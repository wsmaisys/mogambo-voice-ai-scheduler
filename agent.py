# Standard library imports
import os
import re
import json
import logging
import threading
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, TypedDict, Any, Tuple
from collections import Counter

# Third-party imports
import pytz
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END, START
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from logging_config import configure_logging

# Load environment variables here too so the agent can be imported or tested directly.
load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

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
    previous_collected_info: Dict[str, Any] # For tracking changes


# --- Global Cache System ---
class GlobalCache:
    """Thread-safe global cache for optimizing performance and managing state across sessions."""
    _instance: Optional['GlobalCache'] = None
    _cache: Dict[str, Tuple[Any, datetime, timedelta]] = {}
    _lock = threading.Lock()
    DEFAULT_CACHE_AGE = timedelta(minutes=5)
    SESSION_CACHE_AGE = timedelta(hours=24)

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
                value, timestamp, max_age = self._cache[key]
                if datetime.now() - timestamp < max_age:
                    return value
                else:
                    del self._cache[key]  # Expired entry
            return None

    def set(self, key: str, value: Any, max_age: Optional[timedelta] = None) -> None:
        """Stores a value in the cache with the current timestamp."""
        with self._lock:
            ttl = max_age or self.DEFAULT_CACHE_AGE
            self._cache[key] = (value, datetime.now(), ttl)
            self._cleanup()

    def set_session(self, key: str, value: Any) -> None:
        """Stores a user session with a longer session-oriented TTL."""
        self.set(key, value, self.SESSION_CACHE_AGE)

    def delete(self, key: str) -> None:
        """Deletes a cache entry if it exists."""
        with self._lock:
            self._cache.pop(key, None)

    def cleanup_expired(self) -> None:
        """Removes expired entries without invalidating active sessions."""
        with self._lock:
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
        expired_keys = [k for k, (_, t, max_age) in self._cache.items() if now - t > max_age]
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
            return cached_response

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

def find_events_by_summary_and_date(service: Any, calendar_id: str, summary: str, date_str: str) -> List[Dict]:
    """
    Finds events by fuzzy matching summary and date within a specified calendar.
    """
    try:
        # Convert date string to ISO format for the full day
        start_time = f"{date_str}T00:00:00+05:30"
        end_time = f"{date_str}T23:59:59+05:30"

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
            if ratio > 70:  # Slightly more lenient threshold
                matches.append(event)
                logger.debug("event summary matched", extra={"summary": event_summary, "match_ratio": ratio})
        return matches
    except Exception as e:
        logger.exception("event lookup by summary/date failed")
        return []

def parse_datetime_flexible(date_str: str, relative_to: Optional[datetime] = None) -> str:
    """
    Simple and direct datetime parser that handles common cases.
    Returns an ISO 8601 formatted string suitable for Google Calendar API.
    """
    try:
        tz = pytz.timezone('Asia/Kolkata')
        now = relative_to if relative_to else datetime.now(tz)
        today = now.date()
        tomorrow = (now + timedelta(days=1)).date()

        # Default time settings
        hour = 9
        minute = 0

        # Extract date
        if "tomorrow" in date_str.lower():
            target_date = tomorrow
        elif "today" in date_str.lower():
            target_date = today
        else:
            # Try to parse explicit date
            try:
                # First try common formats
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%b %d", "%d %B"]:
                    try:
                        parsed_date = datetime.strptime(date_str.split(' at ')[0].strip(), fmt)
                        target_date = parsed_date.date()
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matches, default to today
                    target_date = today
            except Exception:
                target_date = today

        # Extract time
        if ":" in date_str:  # HH:MM format
            time_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', date_str.lower())
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                ampm = time_match.group(3)
                if ampm == 'pm' and hour != 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
        else:  # Simple hour format or time of day
            hour_match = re.search(r'(\d{1,2})\s*(am|pm)', date_str.lower())
            if hour_match:
                hour = int(hour_match.group(1))
                if hour_match.group(2) == 'pm' and hour != 12:
                    hour += 12
                elif hour_match.group(2) == 'am' and hour == 12:
                    hour = 0
            else:
                # Time of day references
                if "morning" in date_str.lower():
                    hour = 9
                elif "afternoon" in date_str.lower():
                    hour = 14
                elif "evening" in date_str.lower():
                    hour = 18
                elif "night" in date_str.lower():
                    hour = 20

        # Combine date and time
        result = datetime.combine(target_date, time(hour, minute))
        return tz.localize(result).isoformat()

    except Exception as e:
        logger.exception("datetime parsing failed", extra={"date_input": date_str})
        # Return current time as fallback
        return datetime.now(tz).isoformat()

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
        logger.warning("structured LLM output parsing failed", extra={"raw_llm_output": content})
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
        logger.exception("structured response generation failed")
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

# --- Node Functions ---
def detect_mode(state: AgentState) -> AgentState:
    """Initializes flow control flags."""
    logger.debug("detect_mode")
    state['skip_to_response'] = False
    state['is_general_conversation'] = False
    return state
def intelligent_intent_analysis_node(state: AgentState) -> AgentState:
    """
    Analyzes user intent using an LLM, determines the appropriate tool,
    and extracts necessary arguments. Updates the state with analysis results.
    """
    user_input = state['user_input']

    # Store current collected_info to track changes later
    state['previous_collected_info'] = state.get('collected_info', {}).copy()

    # Build comprehensive context for the LLM
    context = context_manager.build_enhanced_context(state)
    history_context = context.get('conversation', 'No history.')

    logger.debug("intent analysis started", extra={
        "session_id": state.get("session_id"),
        "active_event_id": state.get("active_event_id"),
        "last_successful_tool": state.get("last_successful_tool"),
        "conversation_history_size": len(state.get("conversation_history", [])),
    })

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

    logger.info("intent analysis completed", extra={
        "session_id": state.get("session_id"),
        "tool": analysis.tool.value,
        "confidence": analysis.confidence,
        "missing_fields": analysis.missing_fields,
        "needs_clarification": state["pending_clarification"],
    })

    return state

# --- Calendar Tool Nodes ---
def _resolve_event_reference(state: AgentState) -> Optional[str]:
    """
    Enhanced smart event resolution with improved context awareness and fuzzy matching.
    Returns event_id if found, None if needs clarification or no match.
    Updates state['event_context'] and state['last_event_matches'].
    """
    service = state.get('service')
    if not service:
        logger.warning("calendar service unavailable for event resolution")
        return None

    event_id: Optional[str] = None
    calendar_id = state.get('collected_info', {}).get('calendar_id', 'primary')
    user_input = state['user_input'].lower()
    
    logger.debug(f"Resolving event reference from input: {user_input}")
    
    # 1. Extract temporal context (today/tomorrow)
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    search_date = now
    
    if "tomorrow" in user_input:
        search_date = now + timedelta(days=1)
        logger.debug(f"Searching for events tomorrow ({search_date.date()})")
    elif "today" in user_input:
        logger.debug(f"Searching for events today ({search_date.date()})")
    
    # 2. Try active event first if user refers to "this event" or it's the last active one
    if state.get('active_event_id'):
        try:
            event = service.events().get(calendarId=calendar_id, eventId=state['active_event_id']).execute()
            if any(ref in user_input for ref in ['this event', 'that event', 'current event']) or \
               fuzz.ratio(event.get('summary', '').lower(), user_input) > 70:
                logger.debug(f"Found matching active event: {event.get('summary')}")
                state['event_context'] = event
                return state['active_event_id']
        except Exception as e:
            logger.debug("active event id invalid", extra={"active_event_id": state.get("active_event_id")})
            state['active_event_id'] = ''

    # 3. Extract any person/summary references
    # Common name patterns in event titles
    name_pattern = r'(?:meeting|call|sync|chat|catch up|appointment)\s+(?:with|for)?\s+(\w+)'
    name_match = re.search(name_pattern, user_input, re.IGNORECASE)
    person_name = name_match.group(1) if name_match else None
    
    if person_name:
        logger.debug(f"Looking for events with person: {person_name}")
        # Search for events on the target date matching the person's name
        matches = find_events_by_summary_and_date(service, calendar_id, person_name, search_date.date().isoformat())
        if matches:
            if len(matches) == 1:
                state['event_context'] = matches[0]
                logger.debug(f"Found single matching event with {person_name}")
                return matches[0]['id']
            else:
                state['last_event_matches'] = matches
                clarification_details = []
                for i, match in enumerate(matches[:3], 1):
                    start = match['start'].get('dateTime', match['start'].get('date'))
                    if isinstance(start, str):
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        formatted_time = start_dt.astimezone(tz).strftime('%I:%M %p')
                    else:
                        formatted_time = "all day"
                    clarification_details.append(f"{i}. '{match.get('summary')}' at {formatted_time}")
                
                state['clarification_context'] = {
                    'type': 'event_selection',
                    'reason': f"Found multiple events with {person_name}",
                    'matches': matches,
                    'details': clarification_details,
                    'suggestion': "Please specify which event you mean by its number or provide more details."
                }
                logger.debug(f"Multiple matches found: {clarification_details}")
                return None

    # 4. Try numeric references if we have last_event_matches
    if state.get('last_event_matches'):
        numeric_refs = {
            'first': 0, 'second': 1, 'third': 2,
            '1st': 0, '2nd': 1, '3rd': 2,
            'last': -1
        }
        for ref, index in numeric_refs.items():
            if ref in user_input:
                try:
                    events = state['last_event_matches']
                    if 0 <= index < len(events) or (index == -1 and events):
                        idx = index if index >= 0 else len(events) + index
                        event = events[idx]
                        state['event_context'] = event
                        logger.debug(f"Found event by numeric reference: {event.get('summary')}")
                        return event['id']
                except IndexError:
                    logger.debug(f"Invalid index {index} for event matches")
                    continue

    # 5. Fallback to contextual search
    try:
        # Get events for the target date
        start_time = search_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        if events:
            best_match = None
            best_score = 0
            
            for event in events:
                summary = event.get('summary', '').lower()
                description = event.get('description', '').lower()
                combined_text = f"{summary} {description}"
                
                # Calculate match score based on multiple factors
                summary_score = fuzz.ratio(user_input, summary)
                combined_score = fuzz.ratio(user_input, combined_text)
                
                final_score = max(summary_score, combined_score * 0.8)  # Prefer summary matches
                
                if final_score > best_score and final_score > 70:
                    best_score = final_score
                    best_match = event
            
            if best_match:
                state['event_context'] = best_match
                logger.debug(f"Found best matching event: {best_match.get('summary')} (score: {best_score})")
                return best_match['id']
    
    except Exception as e:
        logger.exception("contextual event search failed")

    # No event found
    state['clarification_context'] = {
        'type': 'no_matches',
        'reason': 'No matching events found.',
        'searched_context': {
            'date': search_date.date().isoformat(),
            'person': person_name,
            'input': user_input
        },
        'suggestion': "Could you provide more details about the event you're looking for?"
    }
    logger.debug("No event reference resolved")
    return None

def create_calendar_event_node(state: AgentState) -> AgentState:
    """Creates a new calendar event using collected information."""
    logger.debug("📅 Entering create_calendar_event_node...")
    
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
            logger.debug(f"Event created: {state['tool_output']}")
        else:
            state['tool_output'] = "Failed to retrieve event ID after creation."
            state['error_message'] = "Event creation successful but ID not returned."
            logger.warning("calendar event created without returned id")

    except Exception as e:
        state['error_message'] = f"Failed to create event: {str(e)}"
        state['tool_output'] = f"❌ Failed to create event: {str(e)}"
        logger.exception("calendar event creation failed")

    return state

def retrieve_calendar_events_node(state: AgentState) -> AgentState:
    """Retrieves calendar events based on time range and other filters."""
    logger.debug("📋 Entering retrieve_calendar_events_node...")
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
            logger.debug("Using cached events for retrieval.")
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
            logger.debug("Fetched events from calendar API.")

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
                logger.debug(f"Set active event ID: {state['active_event_id']}")

            state['last_event_matches'] = events # Store for potential future reference
            state['last_successful_tool'] = ToolType.RETRIEVE_EVENTS.value
            state['tool_output'] = (
                f"Here's what I found on your calendar:\n\n" +
                "\n".join(event_list_str) +
                f"\n\nTotal: {len(events)} event{'s' if len(events) != 1 else ''}."
            )
            logger.debug(f"Successfully processed {len(events)} events.")

    except Exception as e:
        state['error_message'] = f"Failed to retrieve events: {str(e)}"
        state['tool_output'] = f"❌ Failed to retrieve events: {str(e)}"
        logger.exception("calendar event retrieval failed")

    return state

def update_calendar_event_node(state: AgentState) -> AgentState:
    """Updates an existing calendar event."""
    logger.debug("✏️ Entering update_calendar_event_node...")
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
            logger.debug(f"Event updated: {state['tool_output']}")
        else:
            state['tool_output'] = "No changes detected to update the event. Please specify what you'd like to change."
            logger.debug("No changes requested for event update.")

    except Exception as e:
        state['error_message'] = f"Failed to update event: {str(e)}"
        state['tool_output'] = f"❌ Failed to update event: {str(e)}"
        logger.exception("calendar event update failed")

    return state

def delete_calendar_event_node(state: AgentState) -> AgentState:
    """Deletes a calendar event."""
    logger.debug("🗑️ Entering delete_calendar_event_node...")
    service = state.get('service')
    if not service:
        state['tool_output'] = "Calendar service is not available. Please ensure you are logged in."
        state['error_message'] = "Calendar service unavailable."
        return state

    args = state['collected_info']
    logger.debug(f"delete_calendar_event_node args type: {type(args)}, value: {args}")
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
        logger.debug(f"Event deleted: {state['tool_output']}")

    except Exception as e:
        state['error_message'] = f"Failed to delete event: {str(e)}"
        state['tool_output'] = f"❌ Failed to delete event: {str(e)}"
        logger.exception("calendar event deletion failed")

    return state

def find_freebusy_node(state: AgentState) -> AgentState:
    """Checks calendar availability for a given time range."""
    logger.debug("🔍 Entering find_freebusy_node...")
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
        logger.debug(f"Free/busy check result: {state['tool_output']}")

    except Exception as e:
        state['error_message'] = f"Failed to check availability: {str(e)}"
        state['tool_output'] = f"❌ Failed to check availability: {str(e)}"
        logger.exception("calendar freebusy check failed")

    return state

def generate_intelligent_response_node(state: AgentState) -> AgentState:
    """
    Generates the final conversational response to the user,
    incorporating tool outputs, errors, and context.
    """
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
        logger.debug("response cache hit", extra={"session_id": state.get("session_id")})
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

    logger.info("response generated", extra={
        "session_id": state.get("session_id"),
        "intended_tool": state.get("intended_tool"),
        "has_error": bool(error_msg),
        "needs_clarification": pending_clarification,
    })

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
    graph.add_node('intent_analysis', intelligent_intent_analysis_node)
    graph.add_node('create_event', create_calendar_event_node)
    graph.add_node('retrieve_events', retrieve_calendar_events_node)
    graph.add_node('update_event', update_calendar_event_node)
    graph.add_node('delete_event', delete_calendar_event_node)
    graph.add_node('find_freebusy', find_freebusy_node)
    graph.add_node('generate_response', generate_intelligent_response_node)

    # Define graph entry point
    graph.add_edge(START, 'detect_mode')
    graph.add_edge('detect_mode', 'intent_analysis')

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

    graph.add_edge('generate_response', END)

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
