// Ensure ONNX Runtime and VAD are loaded first (due to CDN scripts in HTML)
document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    const statusVisualizer = document.getElementById('status-visualizer');
    const statusText = document.getElementById('status-text');
    const visualizerText = document.getElementById('visualizer-text'); // Text inside circle
    const transcriptOutput = document.getElementById('transcriptOutput');
    const ttsStatusDiv = document.getElementById('ttsStatus');
    const ttsAudioOutput = document.getElementById('ttsAudioOutput');
    const recordedAudioPlayback = document.getElementById('recordedAudioPlayback');
    const stopConversationButton = document.getElementById('stopConversation');
    ttsAudioOutput.playbackRate = 1.5; // Set playback rate for TTS audio

    // --- State Variables ---
    let vad_web;
    let isVadReady = false;
    let isListening = false;
    let isSpeaking = false; // VAD speaking flag
    let isAssistantSpeaking = false; // TTS speaking flag
    let rawTranscript = '';
    const targetSampleRate = 16000;

    // --- Helper Functions ---
    function setStatus(message, type = 'idle') {
        statusText.textContent = message;
        if (statusVisualizer) { statusVisualizer.className = `status-visualizer ${type}`; }
        if (visualizerText) { // Update text inside circle based on state
             switch(type) {
                case 'listening': visualizerText.textContent = 'Listening'; break;
                case 'speaking': visualizerText.textContent = 'Hearing You'; break;
                case 'processing': visualizerText.textContent = 'Processing'; break;
                case 'ai-speaking': visualizerText.textContent = 'Speaking'; break;
                case 'error': visualizerText.textContent = 'Error'; break;
                default: visualizerText.textContent = 'Idle';
             }
        }
        // Add matching class to status text for color coordination
        if (statusText) { statusText.className = `status-text ${type}`; }
        console.log(`Status [${type}]: ${message}`);
    }
    function setTtsStatus(message, isError = false) {
        ttsStatusDiv.textContent = message;
        console.log(`TTS Status: ${message}`);
    }
    function clearPreviousRun() {
        transcriptOutput.value = ''; rawTranscript = '';
        ttsAudioOutput.removeAttribute('src'); ttsAudioOutput.load();
        recordedAudioPlayback.removeAttribute('src'); recordedAudioPlayback.load();
        setTtsStatus("");
        setStatus("Ready", "idle"); // Set initial ready state
    }

    // --- Core VAD and Audio Processing ---
    async function initializeVAD() {
        setStatus("Initializing VAD...", 'processing');
        try {
            if (typeof ort === 'undefined') { throw new Error("ONNX Runtime (ort) not loaded."); }
            if (typeof vad === 'undefined') { throw new Error("VAD library (vad) not loaded."); }

            vad_web = await vad.MicVAD.new({
                ort: ort, // Pass ORT object
                modelURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad_web-web@0.0.22/dist/silero_vad.onnx",
                ortURL: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js",
                onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/",  
                samplingRate: targetSampleRate,
                // --- Tune These Thresholds and Timings ---
                positiveSpeechThreshold: 0.6, // Lower = more sensitive to start
                negativeSpeechThreshold: 0.45, // Higher = more sensitive to stop
                minSilenceFrames: 6, // Fewer = stop sooner after silence
                // --- End Tuning ---

                onSpeechStart: () => {
                    if (!isListening || isAssistantSpeaking) return; // Don't trigger if AI is speaking (no barge-in)
                    isSpeaking = true;
                    setStatus("Hearing you...", 'speaking');
                },
                onSpeechEnd: (audio) => {
                    if (!isListening || !isSpeaking) return; // Ignore if VAD misfire or wasn't actively listening
                    isSpeaking = false; // Reset speaking flag
                    setStatus("Processing speech...", 'processing');
                    console.log(`Speech ended. VAD audio length: ${audio.length}`);
                    if (audio && audio.length > 0) {
                        const wavBlob = encodeWAV(audio, targetSampleRate);
                        stopListeningTemporarily(); // Pause VAD immediately (stop then send)
                        sendAudioForTranscription(wavBlob);
                    } else {
                        console.warn("onSpeechEnd called empty audio.");
                        if (isListening) setStatus("Listening...", 'listening'); // Go back to listening if still active
                    }
                },
                 onVADMisfire: () => {
                    console.log("VAD misfire detected");
                    if (isListening && !isSpeaking) setStatus("Listening...", 'listening');
                 }
            });
            if (!vad_web) { throw new Error("VAD creation failed."); }

            isVadReady = true;
            setStatus("Ready", 'idle');
            startListening(); // Start listening automatically

        } catch (error) {
            console.error("Failed to initialize VAD:", error);
            setStatus(`VAD Init Error: ${error.message}`, 'error');
            isVadReady = false;
        }
    }

    function startListening() {
        if (!isVadReady || isListening || isAssistantSpeaking) { // Don't listen if not ready, already listening, or AI is speaking
             if(isAssistantSpeaking) console.log("Blocked startListening: Assistant is speaking.");
             else console.log(`Blocked startListening: Ready=${isVadReady}, Listening=${isListening}`);
             return;
        }
        try {
            vad_web.start(); isListening = true;
            setStatus("Listening...", 'listening');
        } catch (error) {
            console.error("Error starting VAD listening:", error);
            setStatus(`Mic/VAD Start Error: ${error.message}`, 'error');
            isListening = false;
        }
    }

    function stopListeningTemporarily() { // Pause while processing backend calls
        if (!isVadReady || !isListening) return;
        vad_web.pause(); isListening = false;
        // Keep status as 'processing'
        console.log("VAD paused while processing.");
    }

    function stopConversation() { // Stop listening and reset everything
        if (!isVadReady) return;
        try { vad_web.destroy(); } catch (e) { console.error("Error destroying VAD", e); }
        isListening = false; isSpeaking = false; isVadReady = false;
        setStatus("VAD stopped.", 'idle'); console.log("VAD destroyed.");
        clearPreviousRun(); // Clear all outputs and reset state
    }

    stopConversationButton.addEventListener('click', stopConversation);



    // --- Transcription, Gemini, TTS Functions ---
    async function sendAudioForTranscription(audioBlob) {
        setStatus('Transcribing audio...', 'processing');
        transcriptOutput.value = ''; rawTranscript = '';
        const formData = new FormData();
        formData.append('file', audioBlob, `vad_recording.wav`);
        console.log(`Sending VAD audio blob (size: ${audioBlob.size}) to /transcribe endpoint.`);
        try {
            const response = await fetch('/transcribe', { method: 'POST', body: formData });
            if (!response.ok) { let e = await response.json().catch(()=>({error:`HTTP ${response.status}`})); throw new Error(e.error || `HTTP ${response.status}`); }
            const result = await response.json();
            if (result.transcript !== undefined) {
                rawTranscript = result.transcript;
                setStatus('Transcription complete. Processing with Gemini...', 'processing');
                processTranscriptWithGemini(rawTranscript);
            } else { throw new Error("Missing transcript"); }
        } catch (error) {
            setStatus(`Transcription failed: ${error.message}`, 'error');
            console.error('Error during transcription fetch:', error);
            setTimeout(startListening, 2000); // Restart listening after error
        }
    }

    async function processTranscriptWithGemini(textToProcess) {
        if (!textToProcess) { setStatus("Empty transcript", "idle"); setTimeout(startListening, 1000); return; }
        setStatus("Thinking with Gemini...", 'processing');
        console.log("Sending to /gemini_process:", textToProcess);
        try {
            const response = await fetch('/gemini_process', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: textToProcess })
            });
            if (!response.ok) { let e = await response.json().catch(()=>({error:`HTTP ${response.status}`})); throw new Error(e.error || `HTTP ${response.status}`); }
            const result = await response.json();
            if (result.processed_text !== undefined) {
                const processedText = result.processed_text;
                transcriptOutput.value = processedText;
                if(processedText.startsWith("Stop")){

                requestAndPlayTTS("Goodbye!");
                stopConversation();
                return;
                }
                setStatus('Synthesizing Speech...', 'processing'); // Update status before TTS
                requestAndPlayTTS(processedText);
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
        if (!textToSpeak) { setTtsStatus('Nothing to speak.', true); setTimeout(startListening, 1000); return; }
        setTtsStatus('Synthesizing speech...');
        ttsAudioOutput.removeAttribute('src'); ttsAudioOutput.load();
        const ttsUrl = `/tts?text=${encodeURIComponent(textToSpeak)}`;
        console.log(`Requesting TTS from: ${ttsUrl}`);
        ttsAudioOutput.src = ttsUrl;

        const playPromise = ttsAudioOutput.play();
        if (playPromise !== undefined) {
            playPromise.then(_ => {
                isAssistantSpeaking = true; // Set flag
                setStatus('AI Speaking...', 'ai-speaking'); // Update main status
                setTtsStatus('Playing...');
                console.log("TTS autoplay started.");
            }).catch(error => {
                isAssistantSpeaking = false; // Reset flag
                console.error("TTS autoplay failed:", error);
                setTtsStatus('Audio ready. Press play.', false);
                setStatus("Ready", "idle"); // Go back to idle if autoplay fails
                // Don't automatically restart listening if user needs to press play
            });
        } else {
             setTtsStatus("Audio requested.");
             setStatus("Ready", "idle"); // Assume needs manual play
             // Don't automatically restart listening
        }

        ttsAudioOutput.onplaying = () => { // Redundant if promise works, but good fallback
             if (!isAssistantSpeaking) {
                 isAssistantSpeaking = true;
                 setStatus('AI Speaking...', 'ai-speaking');
                 setTtsStatus('Playing...');
                 console.log("TTS playback started (onplaying).");
             }
        };
        ttsAudioOutput.onended = () => {
             isAssistantSpeaking = false; // Reset flag
             setTtsStatus('Speech finished.'); console.log("TTS playback finished.");
             setStatus("Ready", "idle"); // Set back to idle before listening
             startListening(); // Restart listening for next turn
        };
        ttsAudioOutput.onerror = (e) => {
            isAssistantSpeaking = false; // Reset flag
            setTtsStatus('Error playing synthesized speech.', true); console.error('Error on TTS audio element:', ttsAudioOutput.error);
            setStatus("Error during playback", "error");
            setTimeout(startListening, 2000); // Attempt to restart listening after error
        };
    }

    // --- WAV Encoding Helpers ---
    function encodeWAV(samples, sampleRate) { const buffer = new ArrayBuffer(44 + samples.length * 2); const view = new DataView(buffer); writeString(view, 0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); writeString(view, 8, 'WAVE'); writeString(view, 12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); writeString(view, 36, 'data'); view.setUint32(40, samples.length * 2, true); floatTo16BitPCM(view, 44, samples); return new Blob([view], { type: 'audio/wav' }); }
    function writeString(view, offset, string) { for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i)); }
    function floatTo16BitPCM(output, offset, input) { for (let i = 0; i < input.length; i++, offset += 2) { const s = Math.max(-1, Math.min(1, input[i])); output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true); } }

    // --- Initial Setup ---
    setStatus("Initializing VAD...", 'processing');
    initializeVAD();

}); // End DOMContentLoaded