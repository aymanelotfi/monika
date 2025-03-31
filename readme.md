# Monika - Your AI Assistant

Monika is an AI-powered assistant that combines speech-to-text (STT), natural language processing (NLP), and text-to-speech (TTS) capabilities. It uses Whisper for transcription, Gemini for text processing, RealtimeTTS for speech synthesis, and Orpheus for expressing emotions during conversations.

## Features

- **Speech-to-Text (STT):** Converts spoken audio into text using OpenAI's Whisper.
- **Natural Language Processing (NLP):** Processes user input with Google Gemini for refined responses.
- **Text-to-Speech (TTS):** Synthesizes natural-sounding speech using RealtimeTTS.
- **Emotional Expression:** Utilizes Orpheus to express emotions during conversations.
- **Voice Activity Detection (VAD):** Automatically detects when the user is speaking.
- **Interactive Web Interface:** A user-friendly interface for seamless interaction.

## Video Demo

Watch Monika in action:

[![Monika AI Assistant Demo](https://img.youtube.com/vi/_vdlT1uJq2k/0.jpg)](https://www.youtube.com/watch?v=_vdlT1uJq2k)

## Requirements

- Python 3.8 or higher

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd my_app
    ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add the following variables:
      ```
      GEMINI_API_KEY=your-gemini-api-key
      ```
## Usage

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Open the web interface:
    - Navigate to `http://localhost:5000` in your browser.

3. Interact with Monika:
    - Speak into your microphone to start a conversation.
    - Monika will transcribe, process, and respond to your input.

## Endpoints

- `/`: Main web interface.
- `/transcribe`: Handles audio transcription.
- `/gemini_process`: Processes text with Gemini.
- `/tts`: Streams synthesized speech.

## Troubleshooting

- **Whisper model not loading:** Ensure the `whisper` library is installed and the model size is supported.
- **TTS issues:** Verify the RealtimeTTS engine is properly configured.
- **Gemini errors:** Check if the API key is valid and the environment variable is set.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Gemini](https://developers.generativeai.google)
- [RealtimeTTS](https://github.com/RealtimeTTS)

## Future Improvements

- **Reduce TTS Latency:** Address latency issues in the text-to-speech model for more fluid conversations.
- **Interruption Handling:** Implement the ability for users to interrupt Monika while she's speaking.
- **Expanded Language Support:** Add support for multiple languages in both STT and TTS modules.
- **Custom Voice Options:** Allow users to select different voices for the assistant.
- **Offline Mode:** Develop capabilities for basic functionality without internet connectivity.


