// Global variables
let audioContext;
let mediaStreamSource;
let analyser;
let recording = false;
let socket;

// Constants
const SAMPLE_RATE = 16000;  // Match your backend sample rate
const chunk_size_ms = 160;  // Made of look_ahead @80 and encoder_step_length @80

// Initialize audio context
function initAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE  // Match your backend sample rate
        });
    }
}

// Find the nearest power of 2
function nearestPowerOf2(value) {
    return Math.pow(2, Math.round(Math.log2(value)));
}

// Connect to WebSocket
function connectWebSocket() {
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    // WebSocket event handlers
    socket.onopen = () => {
        console.log('WebSocket connected');
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data)
        
        if (data.type === 'transcript') {
            // Update transcript
            document.getElementById('transcript').textContent = data.text;
        } else if (data.type === 'error') {
            console.error('Server error:', data.message);
            document.getElementById('status').textContent = 'Error: ' + data.message;
        }
    };
    
    socket.onclose = () => {
        console.log('WebSocket disconnected');
        // Attempt to reconnect
        setTimeout(connectWebSocket, 1000);
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Update volume meter
function updateVolumeMeter() {
    if (!recording || !analyser) return;
    
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
    
    // Calculate volume level (0-100)
    const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
    const volume = Math.min(100, average * 2); // Scale for better visualization
    
    // Update volume bar
    document.getElementById('volumeBar').style.width = volume + '%';
    
    // Continue updating
    requestAnimationFrame(updateVolumeMeter);
}

// Start recording
async function startRecording() {
    try {
        initAudio();
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: { 
                sampleRate: SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        // Create analyzer for volume meter
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        
        // Calculate the preferred buffer size
        const preferredBufferSize = Math.floor(SAMPLE_RATE * chunk_size_ms / 1000) - 1;
        // Get a valid buffer size
        const bufferSize = nearestPowerOf2(preferredBufferSize);
        console.log(`Using buffer size: ${bufferSize}`);
        
        // Create processor node
        const processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        // Connect audio nodes
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
        mediaStreamSource.connect(analyser);
        mediaStreamSource.connect(processorNode);
        processorNode.connect(audioContext.destination);
        
        // Process audio data
        processorNode.onaudioprocess = (event) => {
            if (recording && socket && socket.readyState === WebSocket.OPEN) {
                // Get audio data (float32 values between -1 and 1)
                const inputData = event.inputBuffer.getChannelData(0);
                
                // Convert to Int16 data (-32768 to 32767)
                const int16Data = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    // Scale float32 data to int16 range and clip
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    int16Data[i] = s < 0 ? s * 32768 : s * 32767;
                }
                
                // Send to server
                socket.send(int16Data.buffer);
            }
        };
        
        // Start volume meter
        recording = true;
        updateVolumeMeter();
        
        // Update UI
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('status').textContent = 'Recording...';
        
        // Store nodes for cleanup
        window.processorNode = processorNode;
        window.mediaStreamSource = mediaStreamSource;
        window.stream = stream;
        
    } catch (error) {
        console.error('Error starting recording:', error);
        document.getElementById('status').textContent = 'Error: ' + error.message;
    }
}

// Stop recording
function stopRecording() {
    recording = false;
    
    // Stop microphone access
    if (window.stream) {
        window.stream.getTracks().forEach(track => track.stop());
    }
    
    // Disconnect audio nodes
    if (window.mediaStreamSource) {
        window.mediaStreamSource.disconnect();
    }
    
    if (window.processorNode) {
        window.processorNode.disconnect();
    }
    
    // Update UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('status').textContent = 'Stopped';
    document.getElementById('volumeBar').style.width = '0%';
}

// Clear transcript
function clearTranscript() {
    // Clear the transcript text in the UI
    document.getElementById('transcript').textContent = '';
    document.getElementById('status').textContent = 'Clearing transcript...';
    
    // Call the reset endpoint
    fetch('/reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Reset response:', data);
        if (data.status === 'success') {
            document.getElementById('status').textContent = 'Transcript cleared';
        } else {
            document.getElementById('status').textContent = 'Error: ' + data.message;
        }
        
        // Restore status after a delay
        setTimeout(() => {
            document.getElementById('status').textContent = recording ? 'Recording...' : 'Ready';
        }, 1500);
    })
    .catch(error => {
        console.error('Error resetting transcript:', error);
        document.getElementById('status').textContent = 'Error clearing transcript';
        
        // Restore status after a delay
        setTimeout(() => {
            document.getElementById('status').textContent = recording ? 'Recording...' : 'Ready';
        }, 1500);
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Connect UI elements
    document.getElementById('startBtn').addEventListener('click', startRecording);
    document.getElementById('stopBtn').addEventListener('click', stopRecording);
    document.getElementById('clearTranscriptBtn').addEventListener('click', clearTranscript)
    
    // Connect to WebSocket
    connectWebSocket();
});