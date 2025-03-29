import os
import whisper
import uuid
import logging
import threading
import wave
import io
from queue import Queue
from datetime import datetime

# Gemini Import
import google.generativeai as genai

from dotenv import load_dotenv

# Flask imports
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    Response,
    stream_with_context,
    url_for,
    send_from_directory,
)

# RealtimeTTS imports
from RealtimeTTS import OrpheusEngine, TextToAudioStream

# --- Configuration ---
# General App Config
load_dotenv()  # Load environment variables from .env file
APP_PORT = int(os.environ.get("APP_PORT", 5000))  # Default Flask port
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
UPLOAD_FOLDER = "./uploads"
# Flask needs a secret key for session management, flash messages etc, even if not used heavily here.
SECRET_KEY = os.environ.get("SECRET_KEY", "a-very-secret-development-key")

# Whisper STT Config
MODEL_SIZE = "small"  # Or "tiny", "small", "medium", "large"
ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "ogg",
    "flac",
    "mp4",
    "mpeg",
    "webm",
}  # For file upload check

# RealtimeTTS Config
TTS_ENGINE_NAME = "orpheus"  # Or other supported engines if added


# --- Gemini Configuration ---  # <-- New Section
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Using 1.5 Flash as it's the latest fast model. Adjust if a specific "2.0 Flash" name becomes available.
GEMINI_MODEL_NAME = "gemini-2.0-flash"
# System prompt instructing Gemini
GEMINI_SYSTEM_PROMPT = """You are processing text that will be spoken aloud by a Text-to-Speech (TTS) model.
Refine or respond to the user's text naturally, as if you were speaking.
You can use the following emotive tags to add expressiveness to the TTS output: <laugh>,<chuckle>,<cough>, <sniffle>, <groan>, <yawn>, <gasp>.
Use these tags sparingly and only where appropriate to make the speech sound more natural. Do not explain the tags, just use them within the text.
Keep the response concise and relevant to the user's input.
Do not use the tags at the beginning of a sentence.
Use a lot of emotionals tags in the text, but do not overuse them. Use them only when it makes sense.

If the user asked to stop the conversation, answer "Stop" and do not say anything else.
"""

# Configure the Gemini client (do this once at startup)
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info(
            f"Gemini API configured successfully for model {GEMINI_MODEL_NAME}."
        )
    except Exception as e:
        logging.error(
            f"Error configuring Gemini API: {e}. Gemini features will be disabled."
        )
        GEMINI_API_KEY = None  # Disable Gemini if configuration fails
else:
    logging.warning(
        "GEMINI_API_KEY environment variable not set. Gemini features will be disabled."
    )


# --- Logging Setup ---
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
# --- Ensure Upload Folder Exists ---
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        logging.info(f"Created upload directory: {UPLOAD_FOLDER}")
    except OSError as e:
        logging.error(f"Error creating upload directory {UPLOAD_FOLDER}: {e}")
        # Decide if you want to exit or try to continue

# --- Initialize Flask App ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # Limit upload size (e.g., 300MB)
app.secret_key = SECRET_KEY
app.logger.setLevel(logging.DEBUG)


# --- Load Models and Initialize Engines ---
# Load Whisper Model
whisper_model = None
try:
    app.logger.info(f"Loading Whisper model: {MODEL_SIZE}...")
    whisper_model = whisper.load_model(MODEL_SIZE)
    app.logger.info("Whisper model loaded successfully.")
except Exception as e:
    app.logger.error(f"CRITICAL: Error loading Whisper model ({MODEL_SIZE}): {e}")
    # Consider how to handle this error - maybe exit or run degraded

# Initialize RealtimeTTS Engine
tts_engine = None
tts_stream = None
try:
    app.logger.info(f"Initializing TTS Engine: {TTS_ENGINE_NAME}")
    if TTS_ENGINE_NAME == "orpheus":
        tts_engine = OrpheusEngine()
    else:
        raise ValueError(f"Unsupported TTS engine: {TTS_ENGINE_NAME}")

    tts_stream = TextToAudioStream(tts_engine, muted=True)
    app.logger.info(f"TTS Engine {TTS_ENGINE_NAME} initialized.")

except Exception as e:
    app.logger.error(
        f"CRITICAL: Error initializing TTS engine ({TTS_ENGINE_NAME}): {e}"
    )
    # Consider how to handle this error

# --- TTS State Management ---
# A single queue is sufficient because the semaphore ensures only one TTS task runs
tts_audio_queue = Queue()
# Semaphore to allow only one TTS synthesis operation at a time
play_text_to_speech_semaphore = threading.Semaphore(1)

# --- Helper Functions ---


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_wave_header_for_engine(engine):  # Same as FastAPI version
    """Creates a WAV header based on the TTS engine's stream info."""
    if not engine:
        return b""
    try:
        _, _, sample_rate = engine.get_stream_info()
        num_channels = 1
        sample_width = 2  # Assuming 16-bit audio
        frame_rate = sample_rate

        wav_header = io.BytesIO()
        with wave.open(wav_header, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)
            wav_file.writeframes(b"")  # Write empty frames to finalize header structure
        wav_header.seek(0)
        wave_header_bytes = wav_header.read()
        wav_header.close()
        return wave_header_bytes
    except Exception as e:
        app.logger.error(f"Error creating WAV header: {e}")
        return b""  # Return empty bytes on error


# --- TTS Background Task and Generator ---


def play_text_to_speech_task(
    app_instance,  # Pass the app instance to create context
    stream: TextToAudioStream,
    text: str,
    audio_queue: Queue,
    semaphore: threading.Semaphore,
):
    """Background task to perform TTS synthesis and queue audio chunks."""
    # --- Use app.app_context() to make app.logger work ---
    with app_instance.app_context():
        instance_id = str(uuid.uuid4())[:8]
        # Now app.logger (or current_app.logger) should work correctly
        app.logger.debug(  # Using app is often preferred inside context
            f'[{instance_id}] Starting TTS synthesis background task for: "{text[:200]}..."'
        )

        first_chunk_time = None  # For debugging first chunk latency

        def on_audio_chunk(chunk):
            nonlocal first_chunk_time
            if first_chunk_time is None:
                first_chunk_time = datetime.now()
                # Use app.logger here too if needed
                app.logger.debug(
                    f"[{instance_id}] First TTS chunk received at {first_chunk_time}"
                )
            audio_queue.put(chunk)

        try:
            if not stream:
                raise ValueError("TTS Stream not initialized")

            stream.feed(text)
            play_start_time = datetime.now()
            app.logger.debug(
                f"[{instance_id}] Calling stream.play() at {play_start_time} (blocking)."
            )

            # Make sure the 'muted' parameter is set if you don't want server sound
            stream.play(on_audio_chunk=on_audio_chunk, muted=True)

            app.logger.info(
                f"[{instance_id}] TTS synthesis complete in background task."
            )
            audio_queue.put(None)
        except Exception as e:
            app.logger.error(
                f"[{instance_id}] Error during TTS processing in background task: {e}",
                exc_info=True,  # Include traceback in log
            )
            audio_queue.put(None)
        finally:
            # CRITICAL: Release the semaphore
            semaphore.release()
            app.logger.debug(f"[{instance_id}] Released TTS semaphore.")


def tts_audio_stream_generator(audio_queue: Queue):
    """Generator function to yield audio chunks from the queue for Flask Response."""
    app.logger.debug("Flask TTS audio generator started.")
    # Determine if header is needed (e.g., not ElevenLabs)
    send_wave_headers = True  # Assume WAV unless engine changes
    header_sent = False

    while True:
        # Block and wait for the next chunk from the background thread
        chunk = audio_queue.get()

        if chunk is None:  # Check for the sentinel value
            app.logger.debug("Flask TTS generator received None sentinel, stopping.")
            break  # Exit the loop to end the stream

        if send_wave_headers and not header_sent:
            header = create_wave_header_for_engine(tts_engine)
            if header:
                app.logger.debug("Flask TTS generator yielding WAV header.")
                yield header
            header_sent = True  # Don't send header again

        # app.logger.debug(f"Flask TTS generator yielding audio chunk size {len(chunk)}.")
        yield chunk

    app.logger.debug("Flask TTS audio generator finished.")


# --- Flask Routes ---


@app.route("/favicon.ico")
def favicon():
    # Flask automatically serves static files if the folder exists
    # but an explicit route can be clearer sometimes.
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Handles audio file upload, transcribes using Whisper."""
    # app.logger.info(f"Fuck off")
    time_0 = datetime.now()
    if not whisper_model:
        app.logger.info("Transcription request failed: Whisper model not loaded.")
        return jsonify(
            {"error": "Transcription service unavailable: Model not loaded."}
        ), 503

    if "file" not in request.files:
        app.logger.info("Transcription request failed: No 'file' part in request.")
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        app.logger.info("Transcription request failed: No file selected.")
        return jsonify({"error": "No selected file."}), 400

    # Optional: Check file extension using ALLOWED_EXTENSIONS
    # if file and allowed_file(file.filename):
    if file:  # Proceed even if extension check is skipped or fails initially
        # Use secure_filename for safety, although we generate a UUID-based name anyway
        # original_filename = secure_filename(file.filename) # Keep for app.logger if needed
        file_extension = (
            file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "webm"
        )
        temp_filename = f"recording_{uuid.uuid4()}.{file_extension}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)

        app.logger.info(f"Received audio file: {file.filename}, saving to {filepath}")

        try:
            file.save(filepath)
            app.logger.info(f"File saved successfully to {filepath}")

            # Perform transcription
            app.logger.info(f"Starting transcription for {filepath}...")
            # Use fp16=False if on CPU or experiencing issues
            result = whisper_model.transcribe(filepath, fp16=False)
            transcript_text = result["text"]
            app.logger.info(f"Transcription complete for {filepath}.")
            time_1 = datetime.now()
            time_diff = (time_1 - time_0).total_seconds()
            app.logger.info(f"Transcription processing time: {time_diff:.2f} seconds")
            return jsonify({"transcript": transcript_text})

        except Exception as e:
            app.logger.error(
                f"Error during transcription processing for {file.filename}: {e}",
                exc_info=True,
            )
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    app.logger.info(f"Removed temporary file: {filepath}")
                except OSError as e:
                    app.logger.error(f"Error removing temporary file {filepath}: {e}")
    else:
        # This part might be reached if allowed_file check is enabled and fails
        app.logger.warning(
            f"Transcription request failed: File type not allowed ({file.filename})."
        )
        return jsonify({"error": "File type not allowed."}), 400


@app.route("/gemini_process", methods=["POST"])
def gemini_process_text():
    """Receives text, processes it with Gemini, returns processed text."""
    if not GEMINI_API_KEY:
        app.logger.error("Gemini process request failed: API key not configured.")
        return jsonify(
            {"error": "Gemini processing is not available (API key missing)."}
        ), 503

    req_data = request.get_json()
    if not req_data or "text" not in req_data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    original_text = req_data["text"]
    app.logger.info(f"Received text for Gemini processing: '{original_text[:100]}...'")

    try:
        # Initialize the specific model for generation
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME, system_instruction=GEMINI_SYSTEM_PROMPT
        )

        # Generate content
        # Combine system prompt (implicitly handled by system_instruction) and user text
        response = model.generate_content(original_text)  # Send only user text here

        # Extract the text from the response
        # Handle potential blocks or safety issues if needed
        processed_text = response.text
        app.logger.info(f"Gemini processed text: '{processed_text[:200]}...'")

        return jsonify({"processed_text": processed_text})

    # Handle potential errors from the Gemini API
    except Exception as e:
        app.logger.error(f"Error during Gemini API call: {e}", exc_info=True)
        # Check for specific Gemini exceptions if the library provides them
        # For now, return a generic error
        return jsonify({"error": f"Gemini processing failed: {str(e)}"}), 500


@app.route("/tts")
def text_to_speech():
    """Handles text input and streams back synthesized audio."""
    time_0 = datetime.now()
    text = request.args.get("text", "")
    if not text:
        app.logger.warning("TTS request failed: No 'text' query parameter.")
        return jsonify({"error": "No text provided for TTS."}), 400

    if not tts_engine or not tts_stream:
        app.logger.error("TTS request failed: TTS Engine or Stream not initialized.")
        return jsonify({"error": "Text-to-Speech service unavailable."}), 503

    app.logger.info(f'Received TTS request for text: "{text[:200]}..."')

    # Try to acquire the semaphore without blocking
    if play_text_to_speech_semaphore.acquire(blocking=False):
        app.logger.debug("TTS semaphore acquired.")
        # The background task will put data into the shared tts_audio_queue
        threading.Thread(
            target=play_text_to_speech_task,
            args=(
                app,
                tts_stream,
                text,
                tts_audio_queue,
                play_text_to_speech_semaphore,
            ),
            daemon=True,  # Allows app to exit even if thread hangs (use carefully)
        ).start()

        # Create the generator for this response
        generator = tts_audio_stream_generator(tts_audio_queue)
        media_type = "audio/wav"  # Default, adjust if using engines like elevenlabs
        time_1 = datetime.now()
        time_diff = (time_1 - time_0).total_seconds()
        app.logger.info(f"TTS request processing time: {time_diff:.2f} seconds")
        app.logger.debug(f"Streaming TTS response with mimetype {media_type}")
        # Use stream_with_context for safety, though direct generator might work here too.
        # stream_with_context ensures the generator has access to request context if needed.
        return Response(
            stream_with_context(generator),
            mimetype=media_type,
            headers={
                # Prevent caching of dynamic audio stream
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    else:
        # Service is busy with another TTS request
        app.logger.warning("TTS request rejected: Service busy (semaphore locked).")
        return jsonify(
            {
                "error": "Text-to-Speech service is currently busy processing another request. Please try again shortly."
            }
        ), 503  # Service Unavailable


# --- Run the App ---
if __name__ == "__main__":
    print(f"Starting Flask server on http://localhost:{APP_PORT}")
    print(f"Debug mode: {DEBUG_MODE}")
    if not whisper_model:
        print("WARNING: Whisper model failed to load. Transcription will not work.")
    if not tts_engine:
        print("WARNING: TTS engine failed to load. Speech synthesis will not work.")

    app.run(port=APP_PORT, debug=True)
