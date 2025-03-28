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
from werkzeug.utils import secure_filename

# RealtimeTTS imports
from RealtimeTTS import OrpheusEngine, TextToAudioStream

# --- Configuration ---
# General App Config
APP_PORT = int(os.environ.get("APP_PORT", 5000))  # Default Flask port
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
UPLOAD_FOLDER = "./uploads"
# Flask needs a secret key for session management, flash messages etc, even if not used heavily here.
SECRET_KEY = os.environ.get("SECRET_KEY", "a-very-secret-development-key")

# Whisper STT Config
MODEL_SIZE = "base"  # Or "tiny", "small", "medium", "large"
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
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
# System prompt instructing Gemini
GEMINI_SYSTEM_PROMPT = """You are processing text that will be spoken aloud by a Text-to-Speech (TTS) model called Orpheus.
Refine or respond to the user's text naturally, as if you were speaking.
You can use the following emotive tags to add expressiveness to the TTS output: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>.
Use these tags sparingly and only where appropriate to make the speech sound more natural. Do not explain the tags, just use them within the text.
Keep the response concise and relevant to the user's input."""

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
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

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

# --- Load Models and Initialize Engines ---
# Load Whisper Model
whisper_model = None
try:
    logging.info(f"Loading Whisper model: {MODEL_SIZE}...")
    whisper_model = whisper.load_model(MODEL_SIZE)
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"CRITICAL: Error loading Whisper model ({MODEL_SIZE}): {e}")
    # Consider how to handle this error - maybe exit or run degraded

# Initialize RealtimeTTS Engine
tts_engine = None
tts_stream = None
try:
    logging.info(f"Initializing TTS Engine: {TTS_ENGINE_NAME}")
    if TTS_ENGINE_NAME == "orpheus":
        tts_engine = OrpheusEngine()
        # Add setup for other engines if needed here
    else:
        raise ValueError(f"Unsupported TTS engine: {TTS_ENGINE_NAME}")

    tts_stream = TextToAudioStream(
        tts_engine, muted=True
    )  # Muted because we stream to client
    logging.info(f"TTS Engine {TTS_ENGINE_NAME} initialized.")

except Exception as e:
    logging.error(f"CRITICAL: Error initializing TTS engine ({TTS_ENGINE_NAME}): {e}")
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
        logging.error(f"Error creating WAV header: {e}")
        return b""  # Return empty bytes on error


# --- TTS Background Task and Generator ---


def play_text_to_speech_task(
    stream: TextToAudioStream,
    text: str,
    audio_queue: Queue,
    semaphore: threading.Semaphore,
):
    """Background task to perform TTS synthesis and queue audio chunks."""
    instance_id = str(uuid.uuid4())[:8]
    logging.info(
        f'[{instance_id}] Starting TTS synthesis background task for: "{text[:50]}..."'
    )

    def on_audio_chunk(chunk):
        # This function is called by RealtimeTTS when an audio chunk is ready
        logging.debug(f"[{instance_id}] Received TTS chunk, adding to queue.")
        audio_queue.put(chunk)

    try:
        if not stream:
            raise ValueError("TTS Stream not initialized")
        stream.feed(text)
        logging.debug(f"[{instance_id}] Calling stream.play() (blocking).")
        # stream.play is blocking until synthesis is complete or interrupted.
        stream.play(on_audio_chunk=on_audio_chunk, muted=True)
        logging.info(f"[{instance_id}] TTS synthesis complete in background task.")
        audio_queue.put(None)  # Use None as a sentinel to signal the end of audio
    except Exception as e:
        logging.error(
            f"[{instance_id}] Error during TTS processing in background task: {e}",
            exc_info=True,
        )
        audio_queue.put(None)  # Ensure the generator stops even on error
    finally:
        # CRITICAL: Release the semaphore so other requests can proceed
        semaphore.release()
        logging.debug(f"[{instance_id}] Released TTS semaphore.")


def tts_audio_stream_generator(audio_queue: Queue):
    """Generator function to yield audio chunks from the queue for Flask Response."""
    logging.debug("Flask TTS audio generator started.")
    # Determine if header is needed (e.g., not ElevenLabs)
    send_wave_headers = True  # Assume WAV unless engine changes
    header_sent = False

    while True:
        # Block and wait for the next chunk from the background thread
        chunk = audio_queue.get()

        if chunk is None:  # Check for the sentinel value
            logging.debug("Flask TTS generator received None sentinel, stopping.")
            break  # Exit the loop to end the stream

        if send_wave_headers and not header_sent:
            header = create_wave_header_for_engine(tts_engine)
            if header:
                logging.debug("Flask TTS generator yielding WAV header.")
                yield header
            header_sent = True  # Don't send header again

        logging.debug(f"Flask TTS generator yielding audio chunk size {len(chunk)}.")
        yield chunk

    logging.debug("Flask TTS audio generator finished.")


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
    time_0 = datetime.now()
    if not whisper_model:
        logging.error("Transcription request failed: Whisper model not loaded.")
        return jsonify(
            {"error": "Transcription service unavailable: Model not loaded."}
        ), 503

    if "file" not in request.files:
        logging.warning("Transcription request failed: No 'file' part in request.")
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        logging.warning("Transcription request failed: No file selected.")
        return jsonify({"error": "No selected file."}), 400

    # Optional: Check file extension using ALLOWED_EXTENSIONS
    # if file and allowed_file(file.filename):
    if file:  # Proceed even if extension check is skipped or fails initially
        # Use secure_filename for safety, although we generate a UUID-based name anyway
        # original_filename = secure_filename(file.filename) # Keep for logging if needed
        file_extension = (
            file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "webm"
        )
        temp_filename = f"recording_{uuid.uuid4()}.{file_extension}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)

        logging.info(f"Received audio file: {file.filename}, saving to {filepath}")

        try:
            file.save(filepath)
            logging.info(f"File saved successfully to {filepath}")

            # Perform transcription
            logging.info(f"Starting transcription for {filepath}...")
            # Use fp16=False if on CPU or experiencing issues
            result = whisper_model.transcribe(filepath, fp16=False)
            transcript_text = result["text"]
            logging.info(f"Transcription complete for {filepath}.")
            time_1 = datetime.now()
            time_diff = (time_1 - time_0).total_seconds()
            logging.info(f"Transcription processing time: {time_diff:.2f} seconds")
            return jsonify({"transcript": transcript_text})

        except Exception as e:
            logging.error(
                f"Error during transcription processing for {file.filename}: {e}",
                exc_info=True,
            )
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logging.info(f"Removed temporary file: {filepath}")
                except OSError as e:
                    logging.error(f"Error removing temporary file {filepath}: {e}")
    else:
        # This part might be reached if allowed_file check is enabled and fails
        logging.warning(
            f"Transcription request failed: File type not allowed ({file.filename})."
        )
        return jsonify({"error": "File type not allowed."}), 400


@app.route("/gemini_process", methods=["POST"])
def gemini_process_text():
    """Receives text, processes it with Gemini, returns processed text."""
    if not GEMINI_API_KEY:
        logging.error("Gemini process request failed: API key not configured.")
        return jsonify(
            {"error": "Gemini processing is not available (API key missing)."}
        ), 503

    req_data = request.get_json()
    if not req_data or "text" not in req_data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    original_text = req_data["text"]
    logging.info(f"Received text for Gemini processing: '{original_text[:100]}...'")

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
        logging.info(f"Gemini processed text: '{processed_text[:100]}...'")

        return jsonify({"processed_text": processed_text})

    # Handle potential errors from the Gemini API
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}", exc_info=True)
        # Check for specific Gemini exceptions if the library provides them
        # For now, return a generic error
        return jsonify({"error": f"Gemini processing failed: {str(e)}"}), 500


@app.route("/tts")
def text_to_speech():
    """Handles text input and streams back synthesized audio."""
    time_0 = datetime.now()
    text = request.args.get("text", "")
    if not text:
        logging.warning("TTS request failed: No 'text' query parameter.")
        return jsonify({"error": "No text provided for TTS."}), 400

    if not tts_engine or not tts_stream:
        logging.error("TTS request failed: TTS Engine or Stream not initialized.")
        return jsonify({"error": "Text-to-Speech service unavailable."}), 503

    logging.info(f'Received TTS request for text: "{text[:50]}..."')

    # Try to acquire the semaphore without blocking
    if play_text_to_speech_semaphore.acquire(blocking=False):
        logging.debug("TTS semaphore acquired.")
        # The background task will put data into the shared tts_audio_queue
        threading.Thread(
            target=play_text_to_speech_task,
            args=(tts_stream, text, tts_audio_queue, play_text_to_speech_semaphore),
            daemon=True,  # Allows app to exit even if thread hangs (use carefully)
        ).start()

        # Create the generator for this response
        generator = tts_audio_stream_generator(tts_audio_queue)
        media_type = "audio/wav"  # Default, adjust if using engines like elevenlabs
        time_1 = datetime.now()
        time_diff = (time_1 - time_0).total_seconds()
        logging.info(f"TTS request processing time: {time_diff:.2f} seconds")
        logging.debug(f"Streaming TTS response with mimetype {media_type}")
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
        logging.warning("TTS request rejected: Service busy (semaphore locked).")
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

    # Use host='0.0.0.0' to make it accessible on your network
    app.run(port=APP_PORT)
