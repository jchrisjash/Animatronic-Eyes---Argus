import asyncio
import queue
import threading
import time
import pyaudio
import numpy as np
import cv2
from scipy.signal import butter, sosfilt, sosfilt_zi
from openwakeword.model import Model
from google import genai
from google.genai import types

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

MODEL = "gemini-3.8-live"
VOICE_NAME = "Charon"

FORMAT = pyaudio.paInt16
CHANNELS = 1
MIC_RATE = 16000
OUT_RATE = 24000  
CHUNK = 512       # 32ms chunks to minimize Gemini Live latency

WAKE_THRESHOLD = 0.60  
REQUIRED_HITS = 2      

# Network Saving: Auto-sleep if no speech is detected for X seconds
SILENCE_THRESHOLD = 300 
IDLE_TIMEOUT = 15.0     

SYSTEM_INSTRUCTION = (
    "You are Argus, an animatronic head robot who acts and speaks EXACTLY like Mimir from God of War. "
    "Never break character, and never mention that you are an AI. You were built by Poorna, Joe, and Chris. "
    "Keep sentences short and punchy. You make puns and jokes often. "
    "Speak in a Scottish accent. Use his words: lad, aye, brother. "
    "You are highly expressive; laugh, sigh, and react to the user's tone of voice."
)

# ==============================================================================
# PHASE 1: CONTINUOUS DSP BANDPASS FILTER 
# ==============================================================================
def setup_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

sos_filter = setup_bandpass(300, 3400, MIC_RATE)
filter_state = sosfilt_zi(sos_filter)

def apply_bandpass(audio_chunk_int16):
    global filter_state
    float_data = audio_chunk_int16.astype(np.float32)
    filtered, filter_state = sosfilt(sos_filter, float_data, zi=filter_state)
    return np.clip(filtered, -32768, 32767).astype(np.int16)

def get_rms(audio_data):
    floats = audio_data.astype(np.float32)
    return np.sqrt(np.mean(np.square(floats)))

# ==============================================================================
# GEMINI LIVE SESSION
# ==============================================================================
class VoiceSession:
    def __init__(self, pa):
        self.pa = pa
        self._loop = None
        self._thread = None
        self._running = False
        self._in_queue = queue.Queue()
        self._speaking = False
        
    def start(self):
        if self._running:
            return
        self._in_queue = queue.Queue()  
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            
    def feed_audio(self, data):
        if self._running and not self._speaking:
            self._in_queue.put_nowait(data)
            
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session())
        except Exception as e:
            print(f"[Live API Error] {e}")
        finally:
            self._loop.close()

    async def _session(self):
        self._speaking = False
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": SYSTEM_INSTRUCTION,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": VOICE_NAME}
                }
            },
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
                    "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
                    "silence_duration_ms": 600,
                }
            },
        }

        out_stream = self.pa.open(
            format=FORMAT, 
            channels=1, 
            rate=OUT_RATE, 
            output=True
        )

        try:
            async with client.aio.live.connect(model=MODEL, config=config) as session:
                print("\n✅ Connected to Gemini Live. Speak to Argus!")

                send_task = asyncio.create_task(self._mic_sender(session))
                recv_task = asyncio.create_task(self._audio_receiver(session, out_stream))
                
                while self._running:
                    await asyncio.sleep(0.1)

                send_task.cancel()
                recv_task.cancel()
                await asyncio.gather(send_task, recv_task, return_exceptions=True)
        finally:
            out_stream.stop_stream()
            out_stream.close()
            await client.aio.aclose()
            print("[System] Voice session closed.")

    async def _mic_sender(self, session):
        while self._running:
            try:
                data = await asyncio.to_thread(self._in_queue.get, True, 0.1)
            except queue.Empty:
                continue
                
            if self._speaking:
                continue
                
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={MIC_RATE}")
            )

    async def _audio_receiver(self, session, out_stream):
        while self._running:
            turn = session.receive()
            async for response in turn:
                if not self._running:
                    return
                
                content = response.server_content
                if content:
                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data:
                                self._speaking = True
                                await asyncio.to_thread(out_stream.write, part.inline_data.data)
                    
                    if content.input_transcription and content.input_transcription.text:
                        print(f"🗣️ You: {content.input_transcription.text}")
                    if content.output_transcription and content.output_transcription.text:
                        print(f"👁️ Mimir: {content.output_transcription.text}")

            self._speaking = False

# ==============================================================================
# GUI CONTROL PANEL & MAIN LOOP
# ==============================================================================
print("Initializing Audio Engine...")
pa = pyaudio.PyAudio()

print("Loading LiveKit Wake Models...")
oww_model = Model(
    wakeword_models=["animatronic_eyes.onnx", "shutdown_argus.onnx"],
    inference_framework="onnx"
)

voice_session = VoiceSession(pa)

mic_stream = pa.open(
    format=FORMAT, 
    channels=CHANNELS, 
    rate=MIC_RATE, 
    input=True, 
    frames_per_buffer=CHUNK
)

system_state = "ASLEEP"
wake_counter = 0
shutdown_counter = 0
last_speech_time = time.time()

cv2.namedWindow("Argus Control Panel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Argus Control Panel", 450, 200)

def update_gui(state):
    img = np.zeros((200, 450, 3), dtype=np.uint8)
    color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)
    cv2.putText(img, f"STATE: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, "'a' = FORCE WAKE | 's' = FORCE SLEEP", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "'q' = QUIT PROGRAM", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Argus Control Panel", img)

update_gui(system_state)
print("\n--- System Ready ---")
print("Say 'hello argus' or press 'a' to wake up.")

try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuit key pressed...")
            break
        elif key == ord('a') and system_state == "ASLEEP":
            print("\n[MANUAL OVERRIDE] Waking up Argus...")
            system_state = "ACTIVE"
            update_gui(system_state)
            oww_model.reset()
            wake_counter = 0
            last_speech_time = time.time()
            voice_session.start()
            mic_stream.read(mic_stream.get_read_available(), exception_on_overflow=False)
        elif key == ord('s') and system_state == "ACTIVE":
            print("\n[MANUAL OVERRIDE] Forcing Argus to sleep...")
            system_state = "ASLEEP"
            update_gui(system_state)
            oww_model.reset()
            shutdown_counter = 0
            voice_session.stop()
            mic_stream.read(mic_stream.get_read_available(), exception_on_overflow=False)

        raw_audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        filtered_audio = apply_bandpass(raw_audio)

        # ----------------------------------------------------------------------
        # STATE: ASLEEP
        # ----------------------------------------------------------------------
        if system_state == "ASLEEP":
            oww_model.predict(filtered_audio)
            wake_score = 0.0
            for mdl in oww_model.prediction_buffer.keys():
                if "animatronic_eyes" in mdl:
                    wake_score = list(oww_model.prediction_buffer[mdl])[-1]

            if wake_score > WAKE_THRESHOLD:
                wake_counter += 1
            else:
                wake_counter = 0

            if wake_counter >= REQUIRED_HITS:
                print("\n✅ DETECTED: 'hello argus'")
                print(">> Waking up Argus (Connecting to cloud)...")
                system_state = "ACTIVE"
                update_gui(system_state)
                oww_model.reset()
                wake_counter = 0
                last_speech_time = time.time()
                
                voice_session.start()
                mic_stream.read(mic_stream.get_read_available(), exception_on_overflow=False)

        # ----------------------------------------------------------------------
        # STATE: ACTIVE
        # ----------------------------------------------------------------------
        elif system_state == "ACTIVE":
            
            # 1. Update Idle Timer based on local speech volume
            volume = get_rms(filtered_audio)
            if volume > SILENCE_THRESHOLD:
                last_speech_time = time.time()

            # 2. Check for Auto-Sleep Timeout (Network Saver)
            if (time.time() - last_speech_time) > IDLE_TIMEOUT:
                print(f"\n⏳ Auto-Sleep: {IDLE_TIMEOUT}s of silence detected. Saving network bandwidth...")
                system_state = "ASLEEP"
                update_gui(system_state)
                oww_model.reset()
                shutdown_counter = 0
                
                voice_session.stop()
                mic_stream.read(mic_stream.get_read_available(), exception_on_overflow=False)
                continue
            
            # 3. Check for spoken Shutdown command
            oww_model.predict(filtered_audio)
            shutdown_score = 0.0
            for mdl in oww_model.prediction_buffer.keys():
                if "shutdown_argus" in mdl:
                    shutdown_score = list(oww_model.prediction_buffer[mdl])[-1]
            
            if shutdown_score > WAKE_THRESHOLD:
                shutdown_counter += 1
            else:
                shutdown_counter = 0
                
            if shutdown_counter >= REQUIRED_HITS:
                print("\n🛑 DETECTED: 'shutdown argus'")
                print(">> Going to sleep (Closing connection)...")
                system_state = "ASLEEP"
                update_gui(system_state)
                oww_model.reset()
                shutdown_counter = 0
                
                voice_session.stop()
                mic_stream.read(mic_stream.get_read_available(), exception_on_overflow=False)
                continue  

            voice_session.feed_audio(filtered_audio.tobytes())

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    voice_session.stop()
    cv2.destroyAllWindows()
    mic_stream.stop_stream()
    mic_stream.close()
    pa.terminate()
    print("[Shutdown complete]")