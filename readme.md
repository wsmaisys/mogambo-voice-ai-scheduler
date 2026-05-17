# 🗓️ Mogambo Voice AI Scheduler

Mogambo Voice AI Scheduler is a FastAPI web app for managing a user's Google Calendar through a chat interface. Users can type requests or record browser audio. The backend transcribes uploaded audio, uses a Mistral-powered LangGraph workflow to identify the intended calendar action, calls the Google Calendar API, and returns a conversational response.

## 🌐 Deployed App

The app is deployed here:

```text
https://mogambo-calendar-assistant.onrender.com/
```

Deployment is configured to run again automatically when changes are pushed to GitHub.

## 🔐 Security & Privacy Priority

Security and privacy are treated as core priorities for this prototype. The app uses Google OAuth for calendar access, keeps session state in memory, and clears the current session data on logout. The frontend also attempts to call logout when the browser window or tab is closed.

Because this is a portfolio prototype, it does not include enterprise-grade controls such as a persistent encrypted session store, centralized audit logging, or managed secret rotation. For production use, deploy with HTTPS, strong environment secrets, restricted OAuth test users/scopes, and a persistent session strategy appropriate for the hosting environment.

## ✅ What This App Currently Does

- Google OAuth login for Calendar access.
- Calendar chat assistant for:
  - creating events
  - retrieving events
  - updating events
  - deleting events
  - checking free/busy availability
  - answering general conversational messages
- Browser voice input through `MediaRecorder`.
- Backend audio conversion/transcription with `pydub` and `SpeechRecognition`.
- Optional browser speech playback using the Web Speech API.
- Calendar display in the authenticated chat page using FullCalendar.
- In-memory per-session conversation state.
- JSON structured logging through `logging_config.py`.

## ⚠️ What This App Does Not Currently Provide

- No persistent database.
- No durable audit log.
- No automated test suite in this repository.
- Auto-deployment is handled by the hosting setup rather than workflow files in this repository.
- No guarantee of zero data retention; session data is held in memory while active and cleared on logout or expiry.
- No production-grade secret management by itself; secrets must be provided through environment variables.

## 🧱 Architecture

- `app.py`: FastAPI app, Google OAuth flow, session handling, chat/voice APIs, calendar utility APIs, and lifespan cleanup task.
- `agent.py`: LangGraph workflow for intent analysis, calendar tool routing, context handling, and response generation.
- `websocket_handler.py`: WebSocket audio transcription helper.
- `static/index.html`: Login/welcome page.
- `static/chat.html`: Calendar and chat UI.
- `logging_config.py`: JSON logging formatter and setup.
- `Dockerfile`: Container build configuration.

## 🔄 Workflow

1. The user logs in with Google OAuth.
2. The app stores the Google Calendar service and conversation state in an in-memory session cache.
3. The user sends text through `/api/chat/stream` or audio through `/api/voice`.
4. Audio uploads are converted to WAV and transcribed with `SpeechRecognition`.
5. `agent.py` uses MistralAI through LangChain to classify the request.
6. LangGraph routes the request to the relevant calendar node:
   - `create_calendar_event`
   - `retrieve_calendar_events`
   - `update_calendar_event`
   - `delete_calendar_event`
   - `find_freebusy`
   - general response or clarification
7. The backend returns a natural-language response to the chat UI.
8. The frontend refreshes the calendar view after successful chat or voice actions.

## 🛠️ Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/wsmaisys/mogambo-voice-ai-scheduler.git
   cd Mogambo-Voice-AI-Scheduler
   ```

2. Install Python dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Configure environment variables. You can start from `.env.example`:

   ```env
   GOOGLE_CLIENT_ID=your-google-oauth-client-id
   GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
   SESSION_SECRET_KEY=replace-with-a-long-random-secret
   MISTRAL_API_KEY=your-mistral-api-key
   DEPLOY_ENV=local
   REDIRECT_URI=http://localhost:8000/auth/callback
   ```

4. In Google Cloud Console, configure the OAuth redirect URI to match `REDIRECT_URI`.

5. Run the server:

   ```sh
   uvicorn app:app --reload
   ```

6. Open the app:

   ```text
   http://localhost:8000
   ```

## 📦 Docker

Build and run the container:

```sh
docker build -t mogambo-voice-ai-scheduler .
docker run --env-file .env -p 8080:8080 mogambo-voice-ai-scheduler
```

The Docker image runs:

```sh
uvicorn app:app --host 0.0.0.0 --port 8080
```

## 🔌 API Routes

- `GET /`: welcome page.
- `GET /login`: starts Google OAuth.
- `GET /auth/callback`: handles Google OAuth callback.
- `GET /chat`: authenticated chat/calendar page.
- `GET /api/chat/stream`: streaming chat endpoint using Server-Sent Events.
- `POST /api/chat`: non-streaming JSON chat endpoint.
- `POST /api/voice`: browser audio upload endpoint.
- `GET /api/user`: current authenticated user info.
- `POST /logout`: clears the current session.
- `GET /health`: health check.
- `GET /api/calendar/events`: fetch calendar events for a time range.
- `GET /api/calendar/availability`: check free/busy availability.

## 🧪 Notes For Local Development

- The app needs valid Google OAuth credentials and a Mistral API key.
- Browser voice recording requires microphone permission.
- Audio conversion requires `ffmpeg`, which is installed in the Docker image. For local development, install `ffmpeg` separately if voice upload conversion fails.
- Sessions are stored in memory, so restarting the server logs users out.
- `LOG_LEVEL` can be set to control structured logging verbosity, for example `LOG_LEVEL=DEBUG`.

## 🔒 Security Notes

- Do not commit `.env`, OAuth client secrets, Mistral keys, or Google credential files.
- Rotate any secret that was previously committed, shared, or exposed.
- Use a strong `SESSION_SECRET_KEY` outside local development.
- Session data is cleared on logout and is not stored in a database.
- For production, use HTTPS and a persistent/session store designed for your deployment environment.

## 📄 License

See [LICENSE](LICENSE).

## 🤝 Credits

Created by Waseem M Ansari at WSMAISYS Lab, homeground. Inspired by the best in AI, privacy, and productivity.
