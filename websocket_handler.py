"""
WebSocket handling for voice recognition.
"""
import io
import logging
from fastapi import WebSocket, WebSocketDisconnect
import speech_recognition as sr
from pydub import AudioSegment
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

async def handle_voice_websocket(websocket: WebSocket, user_sessions, session_id: str):
    """Handle voice recognition WebSocket connection."""
    if not session_id or not user_sessions.get(session_id):
        await websocket.close(code=4001, reason="Unauthorized")
        return
        
    await websocket.accept()
    recognizer = sr.Recognizer()
    audio_data_buffer = b"" # Buffer to accumulate audio chunks
    
    # 16kHz sampling, 16-bit mono = 32000 bytes/sec
    # 2 seconds buffer = 64000 bytes
    PROCESSING_THRESHOLD = 64000 

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_data_buffer += data
            
            if len(audio_data_buffer) >= PROCESSING_THRESHOLD:
                try:
                    # Try WAV first, fallback to WebM
                    try:
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data_buffer), format="wav")
                    except Exception:
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
        logger.info("voice websocket disconnected")
    except Exception as e:
        logger.exception("voice websocket failed")
