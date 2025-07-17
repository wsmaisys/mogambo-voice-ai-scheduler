from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, List, Union, Optional, Dict, Any
import operator
import json
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime, timedelta
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import re
from enum import Enum

load_dotenv()

# Initialize the LLMs
llm_for_intent_and_tool = ChatMistralAI(model_name="mistral-small-latest", temperature=0.1)
llm_for_response = ChatMistralAI(model_name="mistral-small-latest", temperature=0.3)

# --- Pydantic Models for Structured Output ---
class ToolType(str, Enum):
    CREATE_EVENT = "create_calendar_event"
    RETRIEVE_EVENTS = "retrieve_calendar_events"
    UPDATE_EVENT = "update_calendar_event"
    DELETE_EVENT = "delete_calendar_event"
    FIND_FREEBUSY = "find_freebusy"
    NEED_CLARIFICATION = "need_clarification"
    GENERAL_RESPONSE = "general_response"
    NONE = "none"

class IntentAnalysis(BaseModel):
    """Structured output for intent analysis."""
    tool: ToolType = Field(description="The appropriate tool to use")
    complete_args: Dict[str, Any] = Field(default_factory=dict, description="Complete arguments for the tool")
    missing_fields: List[str] = Field(default_factory=list, description="Missing required fields")
    clarification_question: str = Field(default="", description="Question to ask for clarification")
    confidence: float = Field(default=0.0, description="Confidence level (0.0-1.0)")
    reasoning: str = Field(default="", description="Brief reasoning for the decision")

class GeneralResponse(BaseModel):
    """Structured output for general responses."""
    response: str = Field(description="The response text")
    should_continue: bool = Field(default=False, description="Whether to continue the conversation")
    context_preserved: bool = Field(default=True, description="Whether context should be preserved")

# --- Enhanced Agent State Definition ---
class AgentState(TypedDict):
    # Core input/output
    user_input: str
    is_voice_input: bool
    session_id: str
    service: Optional[Any]
    final_response_text: str
    tool_output: str
    error_message: str
    
    # Conversation management
    conversation_history: List[Dict[str, str]]
    pending_clarification: bool
    collected_info: Dict[str, Any]  # Persists across turns
    missing_required_fields: List[str]
    intended_tool: str
    clarification_question: str
    
    # Flow control
    skip_to_response: bool
    is_general_conversation: bool
    analysis_result: Optional[IntentAnalysis]
    
    # Context tracking
    active_event_id: str  # New: Track current event in conversation

# --- Helper Functions ---
def current_time_str() -> str:
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p IST")

def parse_datetime_flexible(date_str: str) -> str:
    """Parse flexible date/time input and return ISO format."""
    try:
        # Handle relative dates
        if "today" in date_str.lower():
            base_date = datetime.now().date()
        elif "tomorrow" in date_str.lower():
            base_date = (datetime.now() + timedelta(days=1)).date()
        elif "next week" in date_str.lower():
            base_date = (datetime.now() + timedelta(weeks=1)).date()
        else:
            # Try to parse as date
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                try:
                    base_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                base_date = datetime.now().date()
        
        # If time is included, extract it
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', date_str)
        if time_match:
            hour, minute, ampm = time_match.groups()
            hour = int(hour)
            minute = int(minute)
            
            if ampm and ampm.upper() == 'PM' and hour != 12:
                hour += 12
            elif ampm and ampm.upper() == 'AM' and hour == 12:
                hour = 0
            
            return f"{base_date}T{hour:02d}:{minute:02d}:00"
        else:
            return f"{base_date}T09:00:00"  # Default to 9 AM
    except Exception:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def find_events_by_summary_and_date(service, calendar_id, summary, date):
    """Find events matching summary/title and date. Returns list of dicts with summary, start, id."""
    from datetime import datetime, time
    import pytz
    tz = pytz.timezone('Asia/Kolkata')
    date_obj = datetime.strptime(date[:10], "%Y-%m-%d")
    time_min = tz.localize(datetime.combine(date_obj, time(0, 0, 0))).isoformat()
    time_max = tz.localize(datetime.combine(date_obj, time(23, 59, 59))).isoformat()
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    matches = []
    for event in events:
        if summary.lower() in event.get('summary', '').lower():
            matches.append({
                'summary': event.get('summary', ''),
                'start': event['start'].get('dateTime', event['start'].get('date')),
                'id': event.get('id', '')
            })
    return matches

def get_structured_response(llm, messages: List[Dict], response_model: BaseModel) -> BaseModel:
    """Get structured response from LLM using Pydantic validation."""
    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract JSON if wrapped in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        # Parse and validate with Pydantic
        return response_model.model_validate_json(content)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Structured output parsing error: {e}")
        # Return fallback response
        if response_model == IntentAnalysis:
            return IntentAnalysis(
                tool=ToolType.NEED_CLARIFICATION,
                clarification_question="I need clarification on your request. Could you please rephrase?",
                reasoning="Failed to parse user intent"
            )
        else:
            return GeneralResponse(
                response="I apologize, but I'm having trouble understanding your request. Could you please rephrase?",
                should_continue=True
            )

# --- Node Functions ---
def detect_mode(state: AgentState) -> AgentState:
    """Determines if the input is voice or text."""
    state['skip_to_response'] = False
    state['is_general_conversation'] = False
    return state

def voice_detection(state: AgentState) -> AgentState:
    """Handles voice input detection and transcription."""
    if not state.get('is_voice_input', False):
        return state
        
    print("🎙️ Voice input detected, listening...")
    
    def listen_to_microphone() -> str:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎙️ Listening...")
            audio = recognizer.listen(source, timeout=10)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio"
        except sr.RequestError as e:
            return f"Error with speech recognition: {e}"
    
    try:
        transcribed_text = listen_to_microphone()
        state['user_input'] = transcribed_text
        print(f"Transcribed: {transcribed_text}")
    except Exception as e:
        print(f"Voice detection error: {e}")
        state['user_input'] = "Voice input failed"
    
    return state

def intelligent_intent_analysis(state: AgentState) -> AgentState:
    """Advanced intent analysis with structured output and context merging."""
    print("🧠 Analyzing user intent...")
    
    user_input = state['user_input']
    history = state.get('conversation_history', [])
    collected = state.get('collected_info', {})
    
    # Build conversation context
    history_context = build_history_context(history, 3)
    
    # Enhanced prompt for better context awareness
    prompt = f"""You are Mogambo, an intelligent calendar assistant created by Waseem M Ansari at WSMAISYS lab.

Context Analysis:
- Current Date/Time: {current_time_str()}
- Conversation History: {history_context}
- Previously Collected Info: {json.dumps(collected, indent=2) if collected else "None"}
- Active Event ID: {state.get('active_event_id', 'None')}
- User Input: "{user_input}"

Your task: Analyze the user's intent and determine the appropriate action.

Calendar Operations Available:
- create_calendar_event: Needs summary, start_time, end_time (ISO format)
- retrieve_calendar_events: Needs time_min, time_max (ISO format)
- update_calendar_event: Needs event_id + fields to update
- delete_calendar_event: Needs event_id
- find_freebusy: Needs time_min, time_max (ISO format)

Special Handling for Event IDs:
1. If user references a previous event (e.g. "that meeting", "the event"), use active_event_id if available
2. If user selects by number (e.g. "the first one"), preserve the numeric reference for later resolution
3. Explicit event IDs are 32-character strings (letters and numbers)

Decision Logic:
1. If user wants general conversation → use "general_response"
2. If calendar operation is clear → use appropriate tool
3. If continuing previous operation → merge new info with collected_info
4. If missing critical info → use "need_clarification"

For date/time parsing:
- "today" = current date
- "tomorrow" = next day
- "next week" = 7 days from now
- Parse relative times intelligently

Respond with JSON matching this structure:
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
    
    try:
        analysis = get_structured_response(llm_for_intent_and_tool, messages, IntentAnalysis)
        
        # Merge new arguments with existing collected_info
        current_collected = state.get('collected_info', {})
        new_collected = analysis.complete_args
        merged_collected = {**current_collected, **new_collected}
        
        # Update state with analysis results
        state['analysis_result'] = analysis
        state['intended_tool'] = analysis.tool.value
        state['collected_info'] = merged_collected  # Use merged data
        state['missing_required_fields'] = analysis.missing_fields
        state['clarification_question'] = analysis.clarification_question
        state['pending_clarification'] = analysis.tool == ToolType.NEED_CLARIFICATION
        state['is_general_conversation'] = analysis.tool == ToolType.GENERAL_RESPONSE
        
        # Set flow control flags
        if analysis.tool in [ToolType.GENERAL_RESPONSE, ToolType.NEED_CLARIFICATION]:
            state['skip_to_response'] = True
        
        print(f"Intent Analysis: {analysis.tool.value} (confidence: {analysis.confidence})")
        print(f"Reasoning: {analysis.reasoning}")
        print(f"Merged Collected Info: {merged_collected}")
        
        return state
        
    except Exception as e:
        print(f"Intent analysis error: {e}")
        state['error_message'] = str(e)
        state['intended_tool'] = ToolType.GENERAL_RESPONSE.value
        state['skip_to_response'] = True
        state['final_response_text'] = "I'm having trouble processing your request. Could you please try again?"
        return state

# --- Enhanced Tool Nodes ---
def create_calendar_event_node(state: AgentState) -> AgentState:
    """Create calendar event with robust validation."""
    print("📅 Creating calendar event...")
    
    try:
        service = state.get('service')
        if not service:
            state['tool_output'] = "Calendar service is not available"
            return state
        
        args = state['collected_info']
        
        # Validate required fields
        required_fields = ['summary', 'start_time', 'end_time']
        missing = [field for field in required_fields if not args.get(field)]
        
        if missing:
            state['tool_output'] = f"Missing required fields: {', '.join(missing)}"
            return state
        
        # Parse and validate times
        try:
            start_time = parse_datetime_flexible(args['start_time'])
            end_time = parse_datetime_flexible(args['end_time'])
        except Exception as e:
            state['tool_output'] = f"Invalid date/time format: {e}"
            return state
        
        # Build event object
        event = {
            'summary': args['summary'],
            'start': {'dateTime': start_time, 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': end_time, 'timeZone': 'Asia/Kolkata'},
        }
        
        # Optional fields
        if args.get('description'):
            event['description'] = args['description']
        if args.get('location'):
            event['location'] = args['location']
        
        # Create the event
        calendar_id = args.get('calendar_id', 'primary')
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event
        ).execute()
        
        # Set active event ID in state
        event_id = created_event.get('id')
        if event_id:
            state['active_event_id'] = event_id
        
        state['tool_output'] = (
            f"✅ Event '{args['summary']}' created successfully!\n"
            f"📅 Start: {start_time}\n"
            f"⏰ End: {end_time}\n"
            f"🔗 Event ID: {event_id}"
        )
        
    except Exception as e:
        print(f"Event creation error: {e}")
        state['error_message'] = str(e)
        state['tool_output'] = f"❌ Failed to create event: {str(e)}"
    
    return state

def retrieve_calendar_events_node(state: AgentState) -> AgentState:
    """Retrieve calendar events with intelligent formatting."""
    print("📋 Retrieving calendar events...")
    
    try:
        service = state.get('service')
        if not service:
            state['tool_output'] = "Calendar service is not available"
            return state
        
        args = state['collected_info']
        
        # Validate time range
        if not args.get('time_min') or not args.get('time_max'):
            state['tool_output'] = "Missing time range for event retrieval"
            return state
        
        # Parse time range and ensure full day coverage in Asia/Kolkata timezone
        try:
            from datetime import datetime, time
            import pytz
            tz = pytz.timezone('Asia/Kolkata')
            # Parse time_min
            date_min = args['time_min'][:10]
            date_obj_min = datetime.strptime(date_min, "%Y-%m-%d")
            time_min_dt = tz.localize(datetime.combine(date_obj_min, time(0, 0, 0)))
            time_min = time_min_dt.isoformat()
            # Parse time_max
            date_max = args['time_max'][:10]
            date_obj_max = datetime.strptime(date_max, "%Y-%m-%d")
            time_max_dt = tz.localize(datetime.combine(date_obj_max, time(23, 59, 59)))
            time_max = time_max_dt.isoformat()
        except Exception as e:
            state['tool_output'] = f"Invalid time range: {e}"
            return state
        
        # Retrieve events
        calendar_id = args.get('calendar_id', 'primary')
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        if not events:
            state['tool_output'] = f"📭 No events found between {time_min} and {time_max}"
        else:
            event_list = []
            for i, event in enumerate(events, 1):
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_id = event.get('id', '')
                event_list.append(f"{i}. {event['summary']} - {start} (ID: {event_id})")
            
            # Store first event ID as active if only one found
            if len(events) == 1:
                state['active_event_id'] = events[0].get('id', '')
            
            state['tool_output'] = (
                f"📅 Found {len(events)} events:\n" + 
                "\n".join(event_list)
            )
        
    except Exception as e:
        print(f"Event retrieval error: {e}")
        state['error_message'] = str(e)
        state['tool_output'] = f"❌ Failed to retrieve events: {str(e)}"
    
    return state

def update_calendar_event_node(state: AgentState) -> AgentState:
    """Update existing calendar event with context-aware ID handling."""
    print("✏️ Updating calendar event...")
    
    try:
        service = state.get('service')
        if not service:
            state['tool_output'] = "Calendar service is not available"
            return state
        
        args = state['collected_info']
        calendar_id = args.get('calendar_id', 'primary')
        
        # Resolve event ID from multiple sources
        event_id = None
        
        # 1. Check if user provided explicit ID
        if 'event_id' in args and args['event_id']:
            event_id = args['event_id']
        
        # 2. Check active event ID from state
        if not event_id and 'active_event_id' in state and state['active_event_id']:
            event_id = state['active_event_id']
        
        # 3. Handle numeric selection from previous response
        if not event_id and state['user_input'].isdigit():
            # Check if we have matches in conversation history
            last_response = next((msg['content'] for msg in reversed(state['conversation_history']) 
                                 if msg['role'] == 'assistant'), "")
            if "events matching your description" in last_response:
                # Extract matches from context
                summary = args.get('summary')
                date = args.get('start_time') or args.get('date')
                if summary and date:
                    matches = find_events_by_summary_and_date(service, calendar_id, summary, date)
                    option_index = int(state['user_input']) - 1
                    if 0 <= option_index < len(matches):
                        event_id = matches[option_index]['id']
                        # Set as active event for future reference
                        state['active_event_id'] = event_id
        
        # If still no ID, search by summary/date
        if not event_id:
            summary = args.get('summary')
            date = args.get('start_time') or args.get('date')
            if summary and date:
                matches = find_events_by_summary_and_date(service, calendar_id, summary, date)
                if matches:
                    options = []
                    for i, ev in enumerate(matches, 1):
                        options.append(f"{i}. {ev['summary']} - {ev['start']} (ID: {ev['id'][:8]}...)")
                    state['tool_output'] = (
                        "I found these events matching your description. Please select one by number:\n" +
                        "\n".join(options)
                    )
                    return state
                else:
                    state['tool_output'] = "No matching events found for your description."
                    return state
            else:
                state['tool_output'] = "Missing event ID for update, and insufficient info to search."
                return state
        
        # Proceed with update if event_id resolved
        existing_event = service.events().get(
            calendarId=calendar_id,
            eventId=event_id
        ).execute()
        
        updates = []
        if args.get('summary'):
            existing_event['summary'] = args['summary']
            updates.append(f"Summary: {args['summary']}")
        if args.get('description'):
            existing_event['description'] = args['description']
            updates.append(f"Description: {args['description']}")
        if args.get('start_time'):
            start_time = parse_datetime_flexible(args['start_time'])
            existing_event['start']['dateTime'] = start_time
            updates.append(f"Start: {start_time}")
        if args.get('end_time'):
            end_time = parse_datetime_flexible(args['end_time'])
            existing_event['end']['dateTime'] = end_time
            updates.append(f"End: {end_time}")
        
        # Apply updates only if there are changes
        if updates:
            updated_event = service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=existing_event
            ).execute()
            state['tool_output'] = (
                f"✅ Event updated successfully!\n"
                f"🔗 Event ID: {event_id}\n"
                f"Updated fields:\n" + "\n".join(updates)
            )
            # Set active event ID
            state['active_event_id'] = event_id
        else:
            state['tool_output'] = "No changes detected to update the event."
    except Exception as e:
        print(f"Event update error: {e}")
        state['error_message'] = str(e)
        state['tool_output'] = f"❌ Failed to update event: {str(e)}"
    return state

def delete_calendar_event_node(state: AgentState) -> AgentState:
    """Delete calendar event with context-aware ID handling."""
    print("🗑️ Deleting calendar event...")
    
    try:
        service = state.get('service')
        if not service:
            state['tool_output'] = "Calendar service is not available"
            return state
        
        args = state['collected_info']
        calendar_id = args.get('calendar_id', 'primary')
        
        # Resolve event ID from multiple sources
        event_id = None
        
        # 1. Check if user provided explicit ID
        if 'event_id' in args and args['event_id']:
            event_id = args['event_id']
        
        # 2. Check active event ID from state
        if not event_id and 'active_event_id' in state and state['active_event_id']:
            event_id = state['active_event_id']
        
        # 3. Handle numeric selection from previous response using last_event_matches
        if not event_id and state['user_input'].isdigit():
            matches = state.get('last_event_matches', [])
            option_index = int(state['user_input']) - 1
            if 0 <= option_index < len(matches):
                event_id = matches[option_index]['id']
        
        # If still no ID, search by summary/date
        if not event_id:
            summary = args.get('summary')
            date = args.get('start_time') or args.get('date')
            if summary and date:
                matches = find_events_by_summary_and_date(service, calendar_id, summary, date)
                if matches:
                    options = []
                    for i, ev in enumerate(matches, 1):
                        options.append(f"{i}. {ev['summary']} - {ev['start']} (ID: {ev['id'][:8]}...)")
                    state['tool_output'] = (
                        "I found these events matching your description. Please select one by number:\n" +
                        "\n".join(options)
                    )
                    return state
                else:
                    state['tool_output'] = "No matching events found for your description."
                    return state
            else:
                state['tool_output'] = "Missing event ID for deletion, and insufficient info to search."
                return state
        
        # Proceed with deletion
        service.events().delete(
            calendarId=calendar_id,
            eventId=event_id
        ).execute()
        
        state['tool_output'] = f"✅ Event deleted successfully"
        # Clear active event after deletion
        if 'active_event_id' in state and state['active_event_id'] == event_id:
            state['active_event_id'] = ""
        
    except Exception as e:
        print(f"Event deletion error: {e}")
        state['error_message'] = str(e)
        state['tool_output'] = f"❌ Failed to delete event: {str(e)}"
    
    return state

def find_freebusy_node(state: AgentState) -> AgentState:
    """Check calendar availability."""
    print("🔍 Checking calendar availability...")
    
    try:
        service = state.get('service')
        if not service:
            state['tool_output'] = "Calendar service is not available"
            return state
        
        args = state['collected_info']
        
        if not args.get('time_min') or not args.get('time_max'):
            state['tool_output'] = "Missing time range for availability check"
            return state
        
        # Parse time range
        try:
            time_min = parse_datetime_flexible(args['time_min'])
            time_max = parse_datetime_flexible(args['time_max'])
        except Exception as e:
            state['tool_output'] = f"Invalid time range: {e}"
            return state
        
        calendar_id = args.get('calendar_id', 'primary')
        
        # Check free/busy
        body = {
            "timeMin": time_min,
            "timeMax": time_max,
            "items": [{"id": calendar_id}]
        }
        
        response = service.freebusy().query(body=body).execute()
        busy_times = response['calendars'][calendar_id]['busy']
        
        if not busy_times:
            state['tool_output'] = f"✅ You are completely free from {time_min} to {time_max}"
        else:
            busy_list = []
            for busy_time in busy_times:
                start = busy_time['start']
                end = busy_time['end']
                busy_list.append(f"• {start} to {end}")
            
            state['tool_output'] = (
                f"⏰ Busy periods found:\n" + 
                "\n".join(busy_list)
            )
        
    except Exception as e:
        print(f"Free/busy check error: {e}")
        state['error_message'] = str(e)
        state['tool_output'] = f"❌ Failed to check availability: {str(e)}"
    
    return state

# --- Helper Functions ---
def build_history_context(history: List[Dict[str, str]], num_exchanges: int = 2) -> str:
    """Builds a string context from the last N exchanges in conversation history."""
    return "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-num_exchanges:]
    )

def add_personality_touch(user_input: str, response_text: str) -> str:
    """Adds a personality touch to the response if needed."""
    if "created by Waseem M Ansari" not in response_text and "introduce" in user_input.lower():
        return f"Hello! I'm Mogambo, your calendar assistant created by Waseem M Ansari at WSMAISYS lab. {response_text}"
    return response_text

def generate_intelligent_response(state: AgentState) -> AgentState:
    """Generate a natural language response with context awareness."""
    user_input = state.get('user_input', '')
    history = state.get('conversation_history', [])
    analysis = state.get('analysis_result', None)
    
    # Build context
    history_context = build_history_context(history, 3) if history else "No previous context"
    
    # Add active event context to prompt
    active_event_context = ""
    if state.get('active_event_id'):
        active_event_context = f"\nCurrent Event Context: {state['active_event_id'][:8]}..."
    
    # Determine response type
    if state.get('is_general_conversation'):
        prompt = (
            f"You are Mogambo, a friendly calendar assistant.\n\n"
            f"User said: \"{user_input}\"\n"
            f"Context: {history_context}{active_event_context}"
        )
    elif state.get('pending_clarification'):
        prompt = (
            f"You are Mogambo, helping clarify a calendar request.\n\n"
            f"User's request: \"{user_input}\"\n"
            f"Analysis: {analysis.clarification_question if analysis else 'Need more information'}\n"
            f"Context: {history_context}{active_event_context}"
        )
    else:
        tool_result = state.get('tool_output', '')
        error_msg = state.get('error_message', '')
        prompt = (
            f"You are Mogambo, providing feedback on a calendar operation.\n\n"
            f"User's request: \"{user_input}\"\n"
            f"Operation result: {tool_result}\n"
            f"Error (if any): {error_msg}\n"
            f"Context: {history_context}{active_event_context}"
        )
    
    messages = [
        {"role": "system", "content": "You are Mogambo, a helpful calendar assistant. Be natural and concise."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_for_response.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = add_personality_touch(user_input, response_text)
        state['final_response_text'] = response_text
        
        # Update conversation history
        state.setdefault('conversation_history', []).extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text}
        ])
        
        print(f"Generated response: {response_text}")
    except Exception as e:
        print(f"Response generation error: {e}")
        state['final_response_text'] = "I've processed your request, but I'm having trouble generating a response."
    
    return state

def text_to_speech_node(state: AgentState) -> AgentState:
    """Optimized TTS."""
    if not state.get('is_voice_input', False):
        return state
    
    try:
        text = state.get('final_response_text', '')
        if not text:
            return state
        
        # Clean text for TTS
        clean_text = re.sub(r'[✅❌📅📭📋⏰🔍🗑️✏️💬🎙️🔊•]', '', text)
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio = AudioSegment.from_file(fp, format="mp3")
        play(audio)
        
    except Exception as e:
        print(f"TTS error: {e}")
    
    return state

def create_intelligent_workflow() -> StateGraph:
    """Create optimized workflow with context tracking."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    nodes = {
        'detect_mode': detect_mode,
        'voice_input': voice_detection,  # Fixed typo in function name
        'intent_analysis': intelligent_intent_analysis,
        'create_event': create_calendar_event_node,
        'retrieve_events': retrieve_calendar_events_node,
        'update_event': update_calendar_event_node,
        'delete_event': delete_calendar_event_node,
        'find_freebusy': find_freebusy_node,
        'generate_response': generate_intelligent_response,
        'text_to_speech': text_to_speech_node,
    }
    
    for name, func in nodes.items():
        graph.add_node(name, func)
    
    # Set up workflow
    graph.add_edge(START, 'detect_mode')
    
    # Voice routing
    graph.add_conditional_edges(
        'detect_mode',
        lambda state: 'voice_input' if state.get('is_voice_input') else 'intent_analysis',
        {'voice_input': 'voice_input', 'intent_analysis': 'intent_analysis'}
    )
    
    graph.add_edge('voice_input', 'intent_analysis')
    
    # Intent-based routing
    def route_after_intent(state: AgentState) -> str:
        if state.get('skip_to_response'):
            return 'generate_response'
        
        tool_mapping = {
            'create_calendar_event': 'create_event',
            'retrieve_calendar_events': 'retrieve_events',
            'update_calendar_event': 'update_event',
            'delete_calendar_event': 'delete_event',
            'find_freebusy': 'find_freebusy'
        }
        
        return tool_mapping.get(state.get('intended_tool', ''), 'generate_response')
    
    graph.add_conditional_edges(
        'intent_analysis',
        route_after_intent,
        {
            'create_event': 'create_event',
            'retrieve_events': 'retrieve_events',
            'update_event': 'update_event',
            'delete_event': 'delete_event',
            'find_freebusy': 'find_freebusy',
            'generate_response': 'generate_response'
        }
    )
    
    # All tools to response
    for tool in ['create_event', 'retrieve_events', 'update_event', 'delete_event', 'find_freebusy']:
        graph.add_edge(tool, 'generate_response')
    
    # Final routing
    graph.add_conditional_edges(
        'generate_response',
        lambda state: 'text_to_speech' if state.get('is_voice_input') else END,
        {'text_to_speech': 'text_to_speech', END: END}
    )
    
    graph.add_edge('text_to_speech', END)
    
    return graph.compile()

# Create workflow
workflow = create_intelligent_workflow()

# State initialization helper
def create_initial_state(
    user_input: str, 
    is_voice: bool = False, 
    session_id: str = "", 
    calendar_service=None,
    previous_state: Optional[AgentState] = None
) -> AgentState:
    """Create initialized state with context carryover"""
    if previous_state:
        # Preserve context from previous interaction
        return AgentState(
            user_input=user_input,
            is_voice_input=is_voice,
            session_id=session_id,
            service=calendar_service or previous_state.get('service'),
            final_response_text="",
            tool_output="",
            error_message="",
            conversation_history=previous_state.get('conversation_history', []),
            pending_clarification=False,
            collected_info=previous_state.get('collected_info', {}),
            missing_required_fields=[],
            intended_tool="",
            clarification_question="",
            skip_to_response=False,
            is_general_conversation=False,
            analysis_result=None,
            active_event_id=previous_state.get('active_event_id', '')
        )
    else:
        # Initial state creation
        return AgentState(
            user_input=user_input,
            is_voice_input=is_voice,
            session_id=session_id,
            service=calendar_service,
            final_response_text="",
            tool_output="",
            error_message="",
            conversation_history=[],
            pending_clarification=False,
            collected_info={},
            missing_required_fields=[],
            intended_tool="",
            clarification_question="",
            skip_to_response=False,
            is_general_conversation=False,
            analysis_result=None,
            active_event_id=""
        )