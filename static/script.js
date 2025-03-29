// Ensure ONNX Runtime and VAD are loaded first (due to CDN scripts in HTML)
document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    const statusDiv = document.getElementById('status');
    const transcriptOutput = document.getElementById('transcriptOutput'); // Shows Gemini/Assistant text
    const ttsStatusDiv = document.getElementById('ttsStatus');
    const ttsAudioOutput = document.getElementById('ttsAudioOutput');
    ttsAudioOutput.playbackRate = 1.5;
    ttsAudioOutput.volume = 1.0;
    const recordedAudioPlayback = document.getElementById('recordedAudioPlayback'); // For debugging user audio
    const stopConv = document.getElementById('stopConv'); // Optional stop button
    // --- State Variables ---
    let vad_web; // VAD processor instance (from vad_web global)
    let isVadReady = false; // Flag indicating VAD models are loaded
    let isListening = false; // Flag indicating VAD is actively processing audio
    let isSpeaking = false; // Flag from VAD callback indicating speech detected
    let rawTranscript = ''; // Stores raw transcript before Gemini
    const targetSampleRate = 16000; // Target sample rate (VAD model expects 16k)

    // --- Helper Functions ---
    function setStatus(message, type = 'idle') { // types: idle, listening, speaking, processing, error
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`; // Update class for styling
        console.log(`Status [${type}]: ${message}`);
    }
    function setTtsStatus(message, isError = false) {
        ttsStatusDiv.textContent = message;
        // Optionally add error class styling for ttsStatusDiv too
        console.log(`TTS Status: ${message}`);
    }
    function clearPreviousRun() {
        transcriptOutput.value = ''; rawTranscript = '';
        ttsAudioOutput.removeAttribute('src'); ttsAudioOutput.load();
        recordedAudioPlayback.removeAttribute('src'); recordedAudioPlayback.load();
        setTtsStatus("");
    }

    // --- Core VAD and Audio Processing ---

    async function initializeVAD() {
        setStatus("Initializing VAD...", 'processing');
        try {
            // Use the global `vad_web` provided by the CDN script
            // Ensure ort is loaded first (it should be if CDN order is correct)
            if (typeof ort === 'undefined') {
                 throw new Error("ONNX Runtime (ort) not loaded. Check CDN script order/URL.");
            }
            if (typeof vad === 'undefined') {
                throw new Error("VAD library (vad) not loaded. Check CDN script order/URL.");
            }

            vad_web = await vad.MicVAD.new({
                // Provide ort object explicitly if needed by the library version
                 ort: ort, // Pass the loaded ONNX runtime

                // Model URL (Using jsdelivr CDN for Silero VAD model)
                modelURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js",
                // ONNX runtime worker URL (improves performance)
                ortURL: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js",
                
                onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/",


                // VAD Thresholds (--- TUNE THESE VALUES ---)
                positiveSpeechThreshold: 0.6, // Confidence threshold to trigger start
                negativeSpeechThreshold: 0.45, // Confidence below this stops speech
                minSilenceFrames: 6,  // How many silent frames confirm end (e.g., ~180ms at 16kHz/32ms frames)
                // Adjust based on how quickly you want it to stop after speech
                // Smaller = faster stop, Larger = more pause tolerance
                // Sampling Rate (must match model)
                samplingRate: targetSampleRate,

                // Callbacks
                onSpeechStart: () => {
                    if (!isListening) return; // Ignore if not actively listening
                    isSpeaking = true;
                    // Note: MicVAD handles internal buffering, we don't need audioBuffer here
                    setStatus("Speech detected...", 'speaking');
                },
                onSpeechEnd: (audio) => { // `audio` is the Float32Array of the detected speech segment
                     if (!isListening || !isSpeaking) return; // Ignore if VAD misfire or wasn't actively listening
                    isSpeaking = false; // Reset speaking flag for next utterance
                    setStatus("Processing speech...", 'processing');
                    console.log(`Speech ended. VAD provided audio data length: ${audio.length}`);

                    // --- Encode the received Float32Array to WAV Blob ---
                     if (audio && audio.length > 0) {
                        // Ensure sampleRate used for encoding matches VAD's rate
                        const wavBlob = encodeWAV(audio, targetSampleRate);
                        console.log(`Encoded WAV Blob size: ${wavBlob.size}`);

                        // --- Optional: Play back the captured audio for debugging ---
                         // const audioUrl = URL.createObjectURL(wavBlob);
                         // recordedAudioPlayback.src = audioUrl;
                         // recordedAudioPlayback.style.display = 'block'; // Show player
                         // recordedAudioPlayback.play();

                        // --- Send for Transcription ---
                        sendAudioForTranscription(wavBlob);

                        // --- Important: Stop listening while processing ---
                        // Prevents immediate re-triggering if TTS starts quickly
                        stopListeningTemporarily();

                    } else {
                        console.warn("onSpeechEnd called but audio data is empty or invalid.");
                        // If listening should continue, reset status
                         if (isListening) setStatus("Listening...", 'listening');
                    }
                },
                 onVADMisfire: () => { // Optional but useful feedback
                    console.log("VAD misfire detected (triggered but likely not speech)");
                    // Reset status if needed, maybe ignore brief misfires
                    if (isListening && !isSpeaking) { // Only reset if not currently in a speaking state
                         setStatus("Listening...", 'listening');
                    }
                 }
            });

            if (!vad_web) { throw new Error("VAD creation failed."); }

            // --- Microphone Permission ---
            // MicVAD requires the stream to be passed if not using its internal mic handling
            // Let's try letting MicVAD handle the mic stream directly for simplicity
            // It should request permission when vad.start() is called if needed.

            isVadReady = true;
            setStatus("Ready. Speak when status shows 'Listening...'", 'idle');
            // --- Start listening automatically after initialization ---
            startListening();

        } catch (error) {
            console.error("Failed to initialize VAD:", error);
            setStatus(`VAD Initialization Error: ${error.message}`, 'error');
            isVadReady = false;
        }
    }

    function startListening() {
        if (!isVadReady) {
            setStatus("VAD not ready.", 'error');
            console.error("Attempted to start listening but VAD is not ready.");
             // Maybe attempt re-initialization after a delay
            return;
        }
        if (isListening) {
            console.warn("Already listening.");
            return;
        }
        try {
            // Start VAD processing (this should also request mic permission if not granted)
            vad_web.start();
            isListening = true;
            setStatus("Listening...", 'listening');
        } catch (error) {
            console.error("Error starting VAD listening:", error);
            setStatus(`Mic/VAD Start Error: ${error.message}`, 'error');
            isListening = false;
        }
    }

    function stopListeningTemporarily() {
        // Used to pause VAD while backend is processing
        if (!isVadReady || !isListening) {
            return; // Nothing to stop
        }
        vad_web.pause(); // Pause VAD processing
        isListening = false;
        // Status is usually 'Processing...' at this point, so don't change it to 'Idle'
        console.log("VAD paused while processing.");
    }

    function stopListeningPermanently() {
         // Could be called by a hypothetical "End Conversation" button
        if (!isVadReady) return;
        vad_web.destroy(); // Release resources
        isListening = false;
        isSpeaking = false;
        isVadReady = false; // Mark as not ready
        setStatus("VAD stopped.", 'idle');
        console.log("VAD destroyed.");
    }


    // --- Transcription, Gemini, TTS (Largely the same, ensure they call startListening on completion/error) ---

    async function sendAudioForTranscription(audioBlob) {
        setStatus('Transcribing audio...', 'processing');
        transcriptOutput.value = ''; rawTranscript = ''; // Clear previous results
        const formData = new FormData();
        formData.append('file', audioBlob, `vad_recording.wav`); // Use WAV extension
        console.log(`Sending VAD audio blob (size: ${audioBlob.size}) to /transcribe endpoint.`);
        try {
            const response = await fetch('/transcribe', { method: 'POST', body: formData });
            if (!response.ok) { let e = await response.json(); throw new Error(e.error || `HTTP ${response.status}`);}
            const result = await response.json();
            if (result.transcript !== undefined) {
                rawTranscript = result.transcript;
                setStatus('Transcription complete. Processing with Gemini...', 'processing');
                console.log("Raw transcript:", rawTranscript);
                processTranscriptWithGemini(rawTranscript); // Call Gemini
            } else { throw new Error("Missing transcript"); }
        } catch (error) {
            setStatus(`Transcription failed: ${error.message}`, 'error');
            console.error('Error during transcription fetch:', error);
            setTimeout(startListening, 2000); // Restart listening after error
        }
    }

    async function processTranscriptWithGemini(textToProcess) {
        if (!textToProcess) { setTimeout(startListening, 1000); return; } // Restart if empty
        setStatus("Sending text to Gemini for processing...", 'processing');
        console.log("Sending to /gemini_process:", textToProcess);
        try {
            const response = await fetch('/gemini_process', { /* ... (POST JSON as before) ... */
                 method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: textToProcess })
             });
            if (!response.ok) { let e = await response.json(); throw new Error(e.error || `HTTP ${response.status}`);}
            const result = await response.json();
            if (result.processed_text !== undefined) {
                const processedText = result.processed_text;
                transcriptOutput.value = processedText;
                setStatus('Gemini processing complete. Synthesizing Speech...', 'processing');
                console.log("Gemini processed text:", processedText);
                requestAndPlayTTS(processedText); // Request TTS
            } else if (result.error) { throw new Error(result.error); }
            else { throw new Error("Missing processed_text"); }
        } catch (error) {
             setStatus(`Gemini processing failed: ${error.message}`, 'error');
             console.error('Error during Gemini processing fetch:', error);
             transcriptOutput.value = `Gemini Error: ${error.message}\n\n(Raw: ${rawTranscript})`;
             setTimeout(startListening, 2000); // Restart listening after error
        }
    }

    function requestAndPlayTTS(textToSpeak) {
        // Autoplay logic remains the same
        if (!textToSpeak) { setTtsStatus('Nothing to speak.', true); setTimeout(startListening, 1000); return; }
        setTtsStatus('Synthesizing speech...');
        ttsAudioOutput.removeAttribute('src'); ttsAudioOutput.load();
        const ttsUrl = `/tts?text=${encodeURIComponent(textToSpeak)}`;
        console.log(`Requesting TTS from: ${ttsUrl}`);
        ttsAudioOutput.src = ttsUrl;

        const playPromise = ttsAudioOutput.play();
        if (playPromise !== undefined) {
            playPromise.then(_ => { setTtsStatus('Speaking...'); console.log("TTS autoplay started."); })
                       .catch(error => { console.error("TTS autoplay failed:", error); setTtsStatus('Audio ready. Press play.', false); setTimeout(startListening, 1000); }); // Start listening even if autoplay fails
        } else { setTtsStatus("Audio requested."); setTimeout(startListening, 1000); } // Start listening as fallback

        ttsAudioOutput.onplaying = () => { if (!ttsStatusDiv.textContent.includes('Speaking')) setTtsStatus('Speaking...'); console.log("TTS playback started."); };
        ttsAudioOutput.onended = () => {
             setTtsStatus('Speech finished.'); console.log("TTS playback finished.");
             // --- Turn-Taking: Restart listening after TTS finishes ---
             startListening(); 
        };
        ttsAudioOutput.onerror = (e) => {
            setTtsStatus('Error playing synthesized speech.', true); console.error('Error on TTS audio element:', ttsAudioOutput.error);
            setTimeout(startListening, 2000); // Restart listening on TTS error
        };
    }
  
    function stopConversation(){
        stopListeningPermanently(); // Stop VAD and clear resources
        setStatus("Conversation ended.", 'idle');
        console.log("Conversation stopped.");
    }

    stopConv.addEventListener('click', stopConversation); // Optional stop button

    // --- WAV Encoding Helper --- (Crucial for sending VAD audio)
    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2); // 44 bytes for header + 16-bit PCM
        const view = new DataView(buffer);
        writeString(view, 0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 1 * 2, true); view.setUint16(32, 1 * 2, true); view.setUint16(34, 16, true);
        writeString(view, 36, 'data'); view.setUint32(40, samples.length * 2, true);
        floatTo16BitPCM(view, 44, samples);
        return new Blob([view], { type: 'audio/wav' });
    }
    function writeString(view, offset, string) { for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i)); }
    function floatTo16BitPCM(output, offset, input) { for (let i = 0; i < input.length; i++, offset += 2) { const s = Math.max(-1, Math.min(1, input[i])); output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true); } }

    // --- Initial Setup ---
    setStatus("Initializing VAD...", 'processing'); // Initial status
    // Initialize VAD when the page is ready
    initializeVAD();

}); // End DOMContentLoaded