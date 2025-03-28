document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements (Keep as before) ---
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusDiv = document.getElementById('status');
    const recordedAudioPlayback = document.getElementById('recordedAudioPlayback');
    const transcriptOutput = document.getElementById('transcriptOutput'); // Will show final processed text
    const speakButton = document.getElementById('speakButton');
    const ttsStatusDiv = document.getElementById('ttsStatus');
    const ttsAudioOutput = document.getElementById('ttsAudioOutput');

    // --- State Variables (Keep as before) ---
    let mediaRecorder;
    let audioChunks = [];
    let audioStream;
    let currentAudioBlob = null;
    let rawTranscript = ''; // Store raw transcript before Gemini processing

    // --- Check Browser Support (Keep as before) ---
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus('Error: getUserMedia not supported on your browser!', true);
        disableRecordingButtons(); disableSpeakButton(); return;
    }

    // --- Helper Functions (Keep setStatus, setTtsStatus, etc. as before) ---
     function setStatus(message, isError = false) {
        statusDiv.textContent = message;
        statusDiv.className = isError ? 'error' : '';
        console.log(`Status: ${message}`);
    }
    function setTtsStatus(message, isError = false) { /* ... keep as before ... */ }
    function disableRecordingButtons() { /* ... keep as before ... */ }
    function disableSpeakButton() { /* ... keep as before ... */ }
    function clearPreviousRun() {
        transcriptOutput.value = ''; // Clear output area
        rawTranscript = ''; // Clear stored transcript
        recordedAudioPlayback.removeAttribute('src');
        recordedAudioPlayback.load();
        ttsAudioOutput.removeAttribute('src');
        ttsAudioOutput.load();
        disableSpeakButton();
        currentAudioBlob = null;
        setStatus("Press 'Start Recording'");
        setTtsStatus("Ready for processed text.");
    }


    // --- Recording Logic (Keep startButton.onclick, stopButton.onclick, etc. as before) ---
    startButton.onclick = async () => {
        clearPreviousRun();
        setStatus('Requesting microphone access...');
        disableRecordingButtons();
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setStatus('Microphone access granted. Initializing recorder...');
            startButton.disabled = true; stopButton.disabled = false; audioChunks = [];
            const mimeTypes = [ 'audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4', 'audio/wav' ];
            let supportedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || '';
            console.log(`Using MIME type: ${supportedMimeType || 'browser default'}`);
            mediaRecorder = new MediaRecorder(audioStream, { mimeType: supportedMimeType });
            mediaRecorder.ondataavailable = event => { if (event.data.size > 0) audioChunks.push(event.data); };
            mediaRecorder.onstop = () => {
                setStatus('Recording stopped. Processing...');
                console.log("MediaRecorder stopped. Chunks collected:", audioChunks.length);
                 if (audioStream) { audioStream.getTracks().forEach(track => track.stop()); audioStream = null; console.log("Microphone stream stopped."); }
                if (!audioChunks || audioChunks.length === 0) { setStatus('Error: No audio data collected.', true); console.error("No audio chunks collected."); startButton.disabled = false; stopButton.disabled = true; return; }
                currentAudioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                console.log(`Audio Blob created. Size: ${currentAudioBlob.size}, Type: ${currentAudioBlob.type}`);
                const audioUrl = URL.createObjectURL(currentAudioBlob);
                recordedAudioPlayback.src = audioUrl; recordedAudioPlayback.hidden = false;
                // --> Change: Now calls transcription, which then calls Gemini processing
                sendAudioForTranscription(currentAudioBlob);

                startButton.disabled = false; stopButton.disabled = true; // Reset buttons after blob created
            };
            mediaRecorder.onerror = (event) => { setStatus(`Recording Error: ${event.error.name}`, true); console.error("MediaRecorder error:", event.error); if (audioStream) { audioStream.getTracks().forEach(track => track.stop()); audioStream = null; } startButton.disabled = false; stopButton.disabled = true; disableSpeakButton(); };
            mediaRecorder.start(1000);
            setStatus('Recording... Press "Stop Recording" to finish.');
        } catch (err) { setStatus(`Error: ${err.name}. Could not access microphone.`, true); console.error("Error accessing microphone:", err); startButton.disabled = false; stopButton.disabled = true; disableSpeakButton(); }
    };
    stopButton.onclick = () => { if (mediaRecorder && mediaRecorder.state === 'recording') { setStatus('Stopping recording...'); mediaRecorder.stop(); } else { console.warn("Stop clicked but recorder not active."); } };


    // --- Transcription Logic (Modified) ---
    async function sendAudioForTranscription(audioBlob) {
        setStatus('Transcribing audio...');
        speakButton.disabled = true;
        transcriptOutput.value = ''; // Clear previous output
        rawTranscript = ''; // Clear previous raw transcript

        const formData = new FormData();
        let fileExtension = (audioBlob.type || 'audio/webm').split('/')[1].split(';')[0];
        const allowedExtensions = ['webm', 'ogg', 'wav', 'mp3', 'mp4', 'm4a', 'flac', 'mpeg'];
        if (!allowedExtensions.includes(fileExtension)) fileExtension = 'webm';
        formData.append('file', audioBlob, `recording.${fileExtension}`);

        console.log(`Sending audio (size: ${audioBlob.size}) to /transcribe endpoint.`);

        try {
            const response = await fetch('/transcribe', { method: 'POST', body: formData });
            if (!response.ok) { // Handle HTTP errors first
                 let errorDetail = `HTTP Error: ${response.status} ${response.statusText}`;
                 try { const errorJson = await response.json(); errorDetail = errorJson.detail || errorJson.error || errorDetail; } catch (e) {}
                 throw new Error(errorDetail);
            }
            const result = await response.json();
            if (result.transcript !== undefined) {
                rawTranscript = result.transcript; // Store the raw transcript
                setStatus('Transcription complete. Processing with Gemini...');
                console.log("Raw transcript:", rawTranscript);
                processTranscriptWithGemini(rawTranscript).then(() => {
                    requestAndPlayTTS(transcriptOutput.value.trim());
                });
            } else {
                throw new Error("Server response missing 'transcript' field.");
            }
        } catch (error) {
            setStatus(`Transcription failed: ${error.message}`, true);
            console.error('Error during transcription fetch:', error);
            transcriptOutput.value = `Transcription Error: ${error.message}`;
            disableSpeakButton();
        }
    }

    // --- NEW: Gemini Processing Logic ---
    async function processTranscriptWithGemini(textToProcess) {
        if (!textToProcess) {
            setStatus("Cannot process empty transcript.", true);
            return;
        }
        setStatus("Sending text to Gemini for processing...");
        console.log("Sending to /gemini_process:", textToProcess);

        try {
            const response = await fetch('/gemini_process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textToProcess })
            });

            if (!response.ok) { // Handle HTTP errors first
                let errorDetail = `HTTP Error: ${response.status} ${response.statusText}`;
                try { const errorJson = await response.json(); errorDetail = errorJson.detail || errorJson.error || errorDetail; } catch (e) {}
                throw new Error(errorDetail);
            }

            const result = await response.json();
            if (result.processed_text !== undefined) {
                const processedText = result.processed_text;
                transcriptOutput.value = processedText; // Display final processed text
                setStatus('Gemini processing complete. Ready to speak.');
                speakButton.disabled = !processedText; // Enable speak button if text exists
                console.log("Gemini processed text:", processedText);
            } else if (result.error) {
                 throw new Error(`Gemini Error: ${result.error}`);
            }
            else {
                 throw new Error("Server response missing 'processed_text' field.");
            }

        } catch (error) {
             setStatus(`Gemini processing failed: ${error.message}`, true);
             console.error('Error during Gemini processing fetch:', error);
             transcriptOutput.value = `Gemini Processing Error: ${error.message}\n\n(Raw transcript was: ${rawTranscript})`; // Show error and raw text
             disableSpeakButton();
        }
    }


    // --- Text-to-Speech Logic (Keep as before) ---
    // This function now uses the text from transcriptOutput, which is the Gemini-processed text
    speakButton.onclick = () => {
        const textToSpeak = transcriptOutput.value.trim(); // Text IS the processed one
        if (!textToSpeak) { setTtsStatus('Nothing to speak.', true); return; }

        setTtsStatus('Synthesizing speech...');
        speakButton.disabled = true;
        ttsAudioOutput.removeAttribute('src'); ttsAudioOutput.load();
        const ttsUrl = `/tts?text=${encodeURIComponent(textToSpeak)}`;
        console.log(`Requesting TTS from: ${ttsUrl}`);
        ttsAudioOutput.src = ttsUrl;
        ttsAudioOutput.play().catch(e => { console.warn("Autoplay failed:", e.message); setTtsStatus("Audio ready. Press play.", false); });

        ttsAudioOutput.onplaying = () => { setTtsStatus('Speaking...'); console.log("TTS playback started."); };
        ttsAudioOutput.onended = () => { setTtsStatus('Speech finished.'); speakButton.disabled = !transcriptOutput.value.trim(); console.log("TTS playback finished."); };
        ttsAudioOutput.onerror = (e) => {
            setTtsStatus('Error playing synthesized speech.', true);
            speakButton.disabled = !transcriptOutput.value.trim();
            console.error('Error on TTS audio element:', ttsAudioOutput.error);
             fetch(ttsUrl).then(async response => { // Attempt to get server error
                 if (!response.ok) { let detail = `Server error: ${response.status}`; try { const errJson = await response.json(); detail = errJson.detail || errJson.error || detail; } catch(e){} console.error(`Detailed server error: ${detail}`); setTtsStatus(`TTS Fetch Error: ${detail}`, true); }
             }).catch(fetchErr => console.error("Error fetching TTS URL directly:", fetchErr));
        };
    };

    async function requestAndPlayTTS(textToSpeak) {
        if (!textToSpeak) {
            setTtsStatus('Nothing to speak.', true);
            return;
        }

        setTtsStatus('Synthesizing speech...');
        ttsAudioOutput.removeAttribute('src'); // Clear previous audio
        ttsAudioOutput.load();

        const ttsUrl = `/tts?text=${encodeURIComponent(textToSpeak)}`;
        console.log(`Requesting TTS from: ${ttsUrl}`);

        // Set the audio source. The browser starts fetching the stream.
        ttsAudioOutput.src = ttsUrl;

        // --- Attempt Autoplay ---
        // We call play() immediately after setting the src.
        // This returns a Promise that resolves if playback starts,
        // and rejects if it's blocked.
        const playPromise = ttsAudioOutput.play();

        if (playPromise !== undefined) {
            playPromise.then(_ => {
                // Autoplay started successfully!
                setTtsStatus('Autoplaying synthesized speech...');
                console.log("TTS autoplay initiated successfully.");
            }).catch(error => {
                // Autoplay was prevented.
                console.error("TTS autoplay failed:", error);
                setTtsStatus('Audio ready. Press play on the player above.', false); // Inform user
                // Optional: You could visually highlight the play button on the audio element here.
            });
        } else {
             // In some older browser scenarios, .play() might not return a promise.
             // Assume playback might start, but provide controls as fallback.
             setTtsStatus("Audio requested. Use player controls if needed.");
        }


        // --- Event listeners for the TTS audio element (Keep these) ---
        ttsAudioOutput.onplaying = () => {
            // Update status only if it wasn't already set by the promise resolution
            if (!ttsStatusDiv.textContent.includes('Autoplaying')) {
                 setTtsStatus('Speaking...');
            }
            console.log("TTS audio playback started (onplaying event).");
        };

        ttsAudioOutput.onended = () => {
            setTtsStatus('Speech finished.');
            console.log("TTS audio playback finished.");
            // No need to re-enable speak button as it's removed
        };

        ttsAudioOutput.onerror = (e) => {
            // Use the existing error handling for playback errors
            setTtsStatus('Error playing synthesized speech.', true);
            console.error('Error on TTS audio element:', ttsAudioOutput.error);
            // (Keep the fetch fallback for detailed server errors)
             fetch(ttsUrl).then(async response => { /* ... */ }).catch(fetchErr => console.error("Error fetching TTS URL directly:", fetchErr));
        };
    }

     recordedAudioPlayback.hidden = true;
     ttsAudioOutput.hidden = false;

}); // End DOMContentLoaded