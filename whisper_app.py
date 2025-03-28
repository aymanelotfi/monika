import os
import whisper

# Use jsonify to send JSON responses
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import logging
import uuid  # To generate unique filenames for recordings

# --- Configuration ---
UPLOAD_FOLDER = "./uploads"
# Add webm/ogg often used by MediaRecorder
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "mp4", "mpeg", "webm"}
MODEL_SIZE = "base"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "super secret key"  # Change this!
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Load Whisper Model ---
try:
    logging.info(f"Loading Whisper model: {MODEL_SIZE}...")
    model = whisper.load_model(MODEL_SIZE)
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Whisper model ({MODEL_SIZE}): {e}")
    model = None


# --- Helper Function ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Ensure Upload Folder Exists ---
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        logging.info(f"Created upload directory: {UPLOAD_FOLDER}")
    except OSError as e:
        logging.error(f"Error creating upload directory {UPLOAD_FOLDER}: {e}")

# --- Routes ---


# Route for handling standard file uploads and displaying the main page
@app.route("/", methods=["GET"])
def index():
    # Initial GET request - just display the page
    if not model:
        flash(
            "Warning: Whisper model failed to load. Transcription is unavailable.",
            "warning",
        )
    return render_template("index.html", transcript=None, error_message=None)


# NEW Route specifically for handling recorded audio via AJAX
@app.route("/transcribe_record", methods=["POST"])
def transcribe_record():
    """Handles audio blob sent from browser recording"""
    if not model:
        # Return JSON error response
        return jsonify({"error": "Whisper model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No audio data received."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Received empty file data."}), 400

    # Although we check ALLOWED_EXTENSIONS for uploads, recordings might have
    # default names like 'blob' or 'recorded_audio.webm'. Whisper is generally
    # good at figuring out the format, so we might skip strict extension check here,
    # but securing the filename is still good practice.
    # We'll generate a unique filename to avoid conflicts.
    file_extension = (
        file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "webm"
    )  # default if no ext
    # Ensure the generated extension is somewhat sensible/allowed if possible
    if file_extension not in ALLOWED_EXTENSIONS:
        logging.warning(
            f"Received recording with potentially unexpected extension: {file_extension}. Proceeding."
        )
        # You might want to default to a common format like 'webm' if unsure
        # file_extension = 'webm' # Or handle as an error depending on strictness

    filename = f"recording_{uuid.uuid4()}.{file_extension}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        logging.info(f"Attempting to save recorded audio to: {filepath}")
        print(f"DEBUG: Attempting to save to: {filepath}")  # Add extra debug print
        print(f"DEBUG: File object: {file}")
        print(f"DEBUG: Content type: {file.content_type}")
        print(
            f"DEBUG: Content length: {file.content_length}"
        )  # Check if data is received

        file.save(filepath)  # <--- This step is apparently failing silently

        logging.info(
            "Recorded audio saved successfully."
        )  # This log might be misleading if save fails silently
        print("DEBUG: File save attempted.")  # Add print after save
        # Use fp16=False for CPU or if encountering issues
        result = model.transcribe(filepath, fp16=False)
        transcript_text = result["text"]
        logging.info(f"Transcription complete for recording {filename}.")

        # Return the transcript as JSON
        return jsonify({"transcript": transcript_text})

    except Exception as e:
        logging.error(f"Error during recording transcription: {e}")
        # Return JSON error response
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        # Clean up the saved recording file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.info(f"Not Removed temporary recording file: {filepath}")
            except OSError as e:
                logging.error(
                    f"Error removing temporary recording file {filepath}: {e}"
                )


# --- Run the App ---
if __name__ == "__main__":
    # Set debug=False for production!
    # host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, port=5001)
