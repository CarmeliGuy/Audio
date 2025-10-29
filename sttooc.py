# # STTS.py
# # Speech-to-Text server with OO design, double-buffer mic streaming, timed windows, and LLM-based text fixing.
#
# import argparse
# import threading
# import time
# import io
# import sys
# import json
# import audioop
# import traceback
#
# import numpy as np
# import pyaudio
# from flask import Flask, jsonify, request
#
# # Optional: start/stop sounds (safe fallback if simpleaudio not installed)
# try:
#     import simpleaudio as sa
# except Exception:
#     sa = None
#
# # Whisper (openai-whisper)
# import whisper
#
# # Your Gemma client layer
# from clientGemma.clientLLM import GemmaConversationBot, GemmaClient
# # python STTS.py --host 127.0.0.1 --port 6003 --window-sec 2.5 --device-index 0 --whisper-model large-v3 --gemma-model gemma-2-2b-it-GGUF
# # curl -X POST http://127.0.0.1:6003/SpeechToTextService/Listen
# # ----------------------------- Default Config -----------------------------
# # ================== prompt for fixing the transcribed text  ==================
#
# DEFAULT_RATE = 44100
# DEFAULT_CHANNELS = 1
# DEFAULT_CHUNK = 1024
# DEFAULT_WINDOW_SEC = 2.5        # length of each capture window before sending to Whisper
# DEFAULT_SILENCE_THRESHOLD = 0   # set >0 if you also want early-stop on silence (optional)
# # DEFAULT_WHISPER_MODEL = "base"
# DEFAULT_WHISPER_MODEL = "large-v3"
# DEFAULT_MODEL_TEMPERATURE = 0.7
# DEFAULT_MODEL_MAX_TOKENS = 4096
#
# DEFAULT_START_SOUND = "sounds/start.wav"
# DEFAULT_STOP_SOUND  = "sounds/stop.wav"
#
# SYSTEM_PROMPT_2_AUDIO = """You are a transcription-refinement assistant.
# Your input is a single raw transcript sentence. Your task is to output only the fully corrected sentence, with any mis-transcribed or ambiguous words replaced by the most contextually appropriate terms.
# Do not add any commentary, explanations, brackets, or questions - just the clean sentence.
#
# steps:
# 1. Scan the sentence for any words that could be wrong (homophones, mis-hearings, garbled tokens).
# 2. Use the entire sentence context and general domain knowledge to pick the single best replacement for each.
# 3. If more than one word seems plausible, choose the one that makes the sentence most natural and coherent.
# 4. Consider that words might have changed in examples: bin->been, trail->tail, clay->claim, cite->site, see->sea, could->would based on the context.
# 5. Preserve the speaker's original style and phrasing wherever possible.
# 6. Output exactly one line: the corrected sentence.
# 7. Keep the sentence structure the same.
#
# Begin.
# """
#
# # ----------------------------- Utilities -----------------------------
# def play_sound(file_path: str):
#     """Play a WAV sound if simpleaudio is available."""
#     if not sa or not file_path:
#         return
#     try:
#         wave_obj = sa.WaveObject.from_wave_file(file_path)
#         play_obj = wave_obj.play()
#         play_obj.wait_done()
#     except Exception as e:
#         print(f"[WARN] Failed to play sound {file_path}: {e}", file=sys.stderr)
#
# def list_input_devices():
#     """Return a list of available input audio devices with indices."""
#     pa = pyaudio.PyAudio()
#     devices = []
#     try:
#         for i in range(pa.get_device_count()):
#             info = pa.get_device_info_by_index(i)
#             if int(info.get("maxInputChannels", 0)) > 0:
#                 devices.append({
#                     "index": i,
#                     "name": info.get("name", ""),
#                     "rate": info.get("defaultSampleRate", 0),
#                     "channels": info.get("maxInputChannels", 0)
#                 })
#     finally:
#         pa.terminate()
#     return devices
#
# def int16_bytes_to_float32_mono(buf: bytes, in_rate: int, out_rate: int = 16000) -> np.ndarray:
#     """
#     Convert raw int16 mono PCM @ in_rate to float32 mono PCM @ out_rate in range [-1, 1].
#     Uses audioop for resampling (no external deps).
#     """
#     if in_rate != out_rate:
#         # audioop.ratecv expects (fragment, width, nchannels, inrate, outrate, state)
#         converted, _ = audioop.ratecv(buf, 2, 1, in_rate, out_rate, None)
#     else:
#         converted = buf
#     # int16 -> float32
#     arr = np.frombuffer(converted, dtype=np.int16).astype(np.float32) / 32768.0
#     return arr
#
# # ----------------------------- Core Classes -----------------------------
# class AudioStream:
#     """
#     Real-time microphone capture with double-buffer windows.
#     Fills a window of 'window_sec' seconds, then flips the active buffer.
#     """
#
#     def __init__(
#         self,
#         rate=DEFAULT_RATE,
#         channels=DEFAULT_CHANNELS,
#         chunk=DEFAULT_CHUNK,
#         window_sec=DEFAULT_WINDOW_SEC,
#         device_index=None,
#         start_sound=DEFAULT_START_SOUND,
#         stop_sound=DEFAULT_STOP_SOUND,
#         silence_threshold=DEFAULT_SILENCE_THRESHOLD
#     ):
#         self.rate = rate
#         self.channels = channels
#         self.chunk = chunk
#         self.window_sec = float(window_sec)
#         self.bytes_per_sample = 2  # paInt16
#         self.device_index = device_index
#         self.silence_threshold = silence_threshold  # 0 to disable
#
#         # Double buffer
#         self.buffers = [bytearray(), bytearray()]
#         self.active = 0
#         self.lock = threading.Lock()
#
#         self.p = None
#         self.stream = None
#         self.running = False
#
#         self.start_sound = start_sound
#         self.stop_sound = stop_sound
#
#     def start(self):
#         if self.running:
#             return
#         self.p = pyaudio.PyAudio()
#         try:
#             self.stream = self.p.open(
#                 format=pyaudio.paInt16,
#                 channels=self.channels,
#                 rate=self.rate,
#                 input=True,
#                 frames_per_buffer=self.chunk,
#                 input_device_index=self.device_index,
#                 stream_callback=self._callback
#             )
#         except Exception as e:
#             if self.p:
#                 self.p.terminate()
#             raise RuntimeError(f"Failed to open input device: {e}")
#
#         # clear buffers
#         with self.lock:
#             self.buffers[0].clear()
#             self.buffers[1].clear()
#             self.active = 0
#
#         self.running = True
#         play_sound(self.start_sound)
#         self.stream.start_stream()
#
#     def stop(self):
#         if not self.running:
#             return
#         try:
#             if self.stream:
#                 self.stream.stop_stream()
#                 self.stream.close()
#         finally:
#             if self.p:
#                 self.p.terminate()
#         self.running = False
#         play_sound(self.stop_sound)
#
#     def _callback(self, in_data, frame_count, time_info, status):
#         try:
#             with self.lock:
#                 self.buffers[self.active].extend(in_data)
#
#                 # Optional: basic RMS (intensity) for silence early-cut (not required)
#                 if self.silence_threshold > 0:
#                     rms = audioop.rms(in_data, 2)  # 2 bytes per sample
#                     # (We could implement an early flip on sustained silence if desired.)
#
#                 # Flip buffer when window filled
#                 bytes_target = int(self.rate * self.window_sec) * self.bytes_per_sample
#                 if len(self.buffers[self.active]) >= bytes_target:
#                     # Flip
#                     self.active = 1 - self.active
#                     # Clean the new active buffer (we'll write to it now)
#                     self.buffers[self.active].clear()
#         except Exception:
#             traceback.print_exc()
#         return (None, pyaudio.paContinue)
#
#     def get_last_complete_window(self) -> bytes:
#         """
#         Returns bytes from the most recently COMPLETED window (the non-active buffer).
#         After flipping, the non-active buffer holds the last full window.
#         This does not block.
#         """
#         with self.lock:
#             idx = 1 - self.active
#             data = bytes(self.buffers[idx])
#             # We do NOT clear here—so caller can pull once; if you want one-shot, you can clear after.
#             return data
#
#     def pull_and_clear_last_window(self) -> bytes:
#         """Get the last completed window and clear it so we don't reuse it."""
#         with self.lock:
#             idx = 1 - self.active
#             data = bytes(self.buffers[idx])
#             self.buffers[idx].clear()
#             return data
#
#
# class WhisperTranscriber:
#     """Transcribe raw float32 mono audio using openai-whisper (no temp files)."""
#
#     def __init__(self, model_name=DEFAULT_WHISPER_MODEL):
#         self.model = whisper.load_model(model_name)
#         print(f"**Whisper model loaded: {model_name}**")
#
#     def transcribe_float32_mono(self, audio_f32: np.ndarray, language='en') -> str:
#         """
#         openai-whisper can accept numpy float32 in [-1, 1] sampled at 16000 Hz.
#         """
#         try:
#             result = self.model.transcribe(audio_f32, language=language)
#             return (result.get("text") or "").strip()
#         except Exception as e:
#             print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
#             return ""
# # ===================================================================
# # =============================================================
# # Optional: Meta SeamlessM4T-v2 Transcriber (Hebrew→English)
# # =============================================================
# class MetaSeamlessTranscriber:
#     """Transcribe or translate speech using Meta SeamlessM4T-v2 model."""
#
#     def __init__(self, model_name="facebook/seamless-m4t-v2-large"):
#         try:
#             from transformers import AutoProcessor, SeamlessM4Tv2Model
#         except ImportError:
#             raise RuntimeError("Please install transformers: pip install transformers soundfile")
#         import soundfile as sf
#
#         self.model_name = model_name
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = SeamlessM4Tv2Model.from_pretrained(model_name)
#         self.sr_target = 16000  # model expects 16 kHz
#         self.sf = sf
#         print(f"**Meta SeamlessM4T-v2 model loaded: {model_name}**")
#
#     def transcribe_float32_mono(self, audio_f32: np.ndarray, language='he') -> str:
#         """
#         Converts float32 mono numpy audio to English text using SeamlessM4T.
#         Hebrew speech → English translation.
#         """
#         try:
#             # Ensure float32 in range [-1, 1]
#             if audio_f32.dtype != np.float32:
#                 audio_f32 = audio_f32.astype(np.float32)
#             audio_f32 = np.clip(audio_f32, -1.0, 1.0)
#
#             # --- Resample to 16k exactly (Meta is very strict) ---
#             int16_audio = (audio_f32 * 32767.0).astype(np.int16)
#             import audioop
#             resampled_bytes, _ = audioop.ratecv(int16_audio.tobytes(), 2, 1, 16000, 16000, None)
#             resampled_f32 = np.frombuffer(resampled_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#
#             # Prepare input for model
#             inputs = self.processor(audios=resampled_f32,
#                                     sampling_rate=16000,
#                                     src_lang="heb",
#                                     tgt_lang="eng",
#                                     return_tensors="pt")
#             output_tokens = self.model.generate(**inputs)
#             text = self.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
#             return text.strip()
#         except Exception as e:
#             print(f"[ERROR] Meta transcriber failed: {e}", file=sys.stderr)
#             return ""
#
#     # def transcribe_float32_mono(self, audio_f32: np.ndarray, language='he') -> str:
#     #     """
#     #     Converts float32 mono numpy audio to English text using SeamlessM4T.
#     #     Hebrew speech → English translation.
#     #     """
#     #     try:
#     #         audio = audio_f32.astype(np.float32)
#     #         # Convert to int16 for compatibility with soundfile-style processors
#     #         inputs = self.processor(audios=audio, sampling_rate=self.sr_target,
#     #                                 src_lang="heb", tgt_lang="eng", return_tensors="pt")
#     #         output_tokens = self.model.generate(**inputs)
#     #         text = self.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
#     #         return text.strip()
#     #     except Exception as e:
#     #         print(f"[ERROR] Meta transcriber failed: {e}", file=sys.stderr)
#     #         return ""
#
# # ===================================================================
# class GemmaFixer:
#     """Wrap your Gemma client to fix/normalize the STT output."""
#
#     def __init__(
#         self,
#         system_message: str = SYSTEM_PROMPT_2_AUDIO,
#         model_name: str = "gemma-2-2b-it-GGUF",
#         temperature: float = DEFAULT_MODEL_TEMPERATURE,
#         max_tokens: int = DEFAULT_MODEL_MAX_TOKENS
#     ):
#         self.client = GemmaClient()
#         self.bot = GemmaConversationBot(
#             client=self.client,
#             system_message=system_message,
#             k=1,
#             model=model_name,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
#
#     def fix(self, raw_text: str) -> str:
#         try:
#             msg = f"fix the following text:\n{raw_text}"
#             return self.bot.UILLM(messegeIn=msg).strip()
#         except Exception as e:
#             print(f"[ERROR] Gemma fixer failed: {e}", file=sys.stderr)
#             return raw_text.strip()
#
#
# class STTService:
#     """
#     High-level service: microphone window capture -> Whisper transcription -> Gemma fix.
#     """
#
#     def __init__(self, audio: AudioStream, transcriber: WhisperTranscriber, fixer: GemmaFixer):
#         self.audio = audio
#         self.transcriber = transcriber
#         self.fixer = fixer
#
#     def capture_once_and_transcribe(self, wait_for_window=True, language='en'):
#         """
#         Start the mic (if not running), wait one window, pull, transcribe, and fix.
#         Returns dict with 'raw', 'fixed', and timing info.
#         """
#         started_here = False
#         if not self.audio.running:
#             self.audio.start()
#             started_here = True
#
#         # Wait until a full window is produced by the callback (flip happens on fill)
#         if wait_for_window:
#             # Allow at least one flip to happen
#             # Worst-case we sleep ~window_sec + small margin
#             time.sleep(self.audio.window_sec + 0.05)
#
#         # Pull and clear last full window
#         window_bytes = self.audio.pull_and_clear_last_window()
#         if not window_bytes:
#             # If nothing recorded yet (e.g., right after start), sleep a tad and try once more
#             time.sleep(self.audio.window_sec)
#             window_bytes = self.audio.pull_and_clear_last_window()
#
#         if started_here:
#             # We stop right after one-shot; for continuous mode, you can keep it running.
#             self.audio.stop()
#
#         if not window_bytes:
#             return {"raw": "", "fixed": "", "error": "No audio captured"}
#
#         # Convert to 16 kHz mono float32 for Whisper
#         audio_f32 = int16_bytes_to_float32_mono(window_bytes, in_rate=self.audio.rate, out_rate=16000)
#
#         # Transcribe
#         raw_text = self.transcriber.transcribe_float32_mono(audio_f32, language=language)
#
#         # Fix with LLM
#         fixed_text =raw_text # self.fixer.fix(raw_text) if raw_text else ""
#
#         return {
#             "raw": raw_text,
#             "fixed": fixed_text,
#             "window_sec": self.audio.window_sec,
#             "rate": self.audio.rate,
#             "device_index": self.audio.device_index
#         }
#
# # ----------------------------- Flask App Factory -----------------------------
# def create_app(stt_service: STTService):
#     app = Flask(__name__)
#
#     @app.route("/health", methods=["GET"])
#     def health():
#         return jsonify({"ok": True})
#
#     @app.route("/devices", methods=["GET"])
#     def devices():
#         return jsonify({"devices": list_input_devices()})
#
#     @app.route("/SpeechToTextService/Listen", methods=["POST"])
#     def listen_and_transcribe():
#         """
#         One-shot API: capture exactly one window (e.g., 2.5s), transcribe+fix, return JSON.
#         """
#         lang = (request.args.get("lang") or "en").strip()
#         try:
#             result = stt_service.capture_once_and_transcribe(wait_for_window=True, language=lang)
#             if result.get("error"):
#                 return jsonify({"error": result["error"]}), 400
#             return jsonify({"transcription": result["fixed"], "raw": result["raw"], "meta": {
#                 "window_sec": result["window_sec"],
#                 "rate": result["rate"],
#                 "device_index": result["device_index"]
#             }}), 200
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"error": str(e)}), 500
#
#     return app
# # ======================================================================
# def build_transcriber(name: str):
#     """
#     Factory: choose between Whisper or Meta SeamlessM4T-v2 for transcription.
#     Use --whisper-model meta or --whisper-model seamless-m4t-v2-large
#     """
#     n = name.lower().strip()
#     if n == "meta" or "seamless" in n or "m4t" in n:
#         return MetaSeamlessTranscriber()
#     return WhisperTranscriber(model_name=name)
# # ======================================================================
# # ======================================================================
# # ----------------------------- Main / CLI -----------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Streaming STT server (mic -> Whisper -> Gemma fixer)")
#     parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind (default: 127.0.0.1)")
#     parser.add_argument("--port", type=int, default=6003, help="Port to bind (default: 6003)")
#     parser.add_argument("--device-index", type=int, default=None, help="Audio input device index (see /devices)")
#     parser.add_argument("--rate", type=int, default=DEFAULT_RATE, help=f"Sampling rate (default: {DEFAULT_RATE})")
#     parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help=f"Channels (default: {DEFAULT_CHANNELS})")
#     parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help=f"Frames per buffer (default: {DEFAULT_CHUNK})")
#     parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help=f"Window seconds per transcript (default: {DEFAULT_WINDOW_SEC})")
#     parser.add_argument("--silence-threshold", type=int, default=DEFAULT_SILENCE_THRESHOLD, help="Optional RMS threshold for silence (0 disables)")
#     parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help=f"Whisper model name (default: {DEFAULT_WHISPER_MODEL})")
#     parser.add_argument("--start-sound", default=DEFAULT_START_SOUND, help="Path to start beep WAV (optional)")
#     parser.add_argument("--stop-sound", default=DEFAULT_STOP_SOUND, help="Path to stop beep WAV (optional)")
#     parser.add_argument("--gemma-model", default="gemma-2-2b-it-GGUF", help="Gemma model id for fixer")
#     parser.add_argument("--debug", action="store_true", help="Flask debug mode")
#
#     args = parser.parse_args()
#
#     # Build objects
#     audio = AudioStream(
#         rate=args.rate,
#         channels=args.channels,
#         chunk=args.chunk,
#         window_sec=args.window_sec,
#         device_index=args.device_index,
#         start_sound=args.start_sound,
#         stop_sound=args.stop_sound,
#         silence_threshold=args.silence_threshold
#     )
#     # transcriber = WhisperTranscriber(model_name=args.whisper_model)
#     transcriber = build_transcriber(args.whisper_model)
#
#     fixer = GemmaFixer(model_name=args.gemma_model, system_message=SYSTEM_PROMPT_2_AUDIO)
#
#     # fixer = GemmaFixer(model_name=args.gemma_model, system_message=systemprompt2audio)
#
#     stt_service = STTService(audio, transcriber, fixer)
#     app = create_app(stt_service)
#
#     print(f"Server running on {args.host}:{args.port}")
#     print("Try:  curl -X POST http://%s:%d/SpeechToTextService/Listen" % (args.host, args.port))
#     app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
#
# if __name__ == "__main__":
#     main()
#
#
#
# # curl -X POST "http://127.0.0.1:6003/SpeechToTextService/Listen?lang=he&fix=false"
# # python STTS.py --host 127.0.0.1 --port 6003 --window-sec 2.5 --device-index 0 --whisper-model large-v3 --gemma-model gemma-2-2b-it-GGUF
# # python STTS.py --host 127.0.0.1 --port 6003 --window-sec 2.5 --device-index 0 --whisper-model meta --gemma-model gemma-2-2b-it-GGUF
#
#
#
# # python STTS.py --host 127.0.0.1 --port 6003 --window-sec 70 --device-index 0 --whisper-model meta --gemma-model gemma-2-2b-it-GGUF
#




# sttooc_fixed.py — Adaptive Speech-to-Text Server with Double Buffering (final corrected)

import argparse, threading, time, sys, traceback, audioop, logging
import numpy as np
import pyaudio
from flask import Flask, jsonify, request
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional

try:
    import simpleaudio as sa
except Exception:
    sa = None

import whisper
from clientGemma.clientLLM import GemmaConversationBot, GemmaClient

# ============================================================
# -------------------------- CONFIG --------------------------
# ============================================================

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6003
DEFAULT_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK = 1024
DEFAULT_DEVICE_INDEX = 0
DEFAULT_SILENCE_THRESHOLD = 500
DEFAULT_SILENCE_DURATION = 2.5          # stop after 2.5s continuous quiet
DEFAULT_BUFFER_SWITCH_SILENCE = 0.5     # flip buffer after 0.5s quiet
DEFAULT_MIN_CHUNK_DURATION = 0.3
DEFAULT_MAX_RECORDING_TIME = 60.0
DEFAULT_HARD_SWITCH_TIME = 70.0
DEFAULT_START_SOUND = "sounds/start.wav"
DEFAULT_STOP_SOUND = "sounds/stop.wav"
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_GEMMA_MODEL = "gemma-2-2b-it-GGUF"
DEFAULT_MODEL_TEMPERATURE = 0.0
DEFAULT_MODEL_MAX_TOKENS = 4096
DEFAULT_DEBUG = False

SYSTEM_PROMPT_2_AUDIO = """You are a transcription-refinement assistant.
Your input is a single raw transcript sentence. Output only the corrected sentence.
Do not add commentary or brackets.
"""

# ============================================================
# -------------------------- LOGGING -------------------------
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# ------------------------ DATA CLASSES ----------------------
# ============================================================

@dataclass
class AudioChunk:
    data: bytes
    buffer_id: int
    timestamp: float
    duration: float

# ============================================================
# -------------------------- UTILITIES -----------------------
# ============================================================

def play_sound(path: str):
    """Play a sound asynchronously."""
    if not sa or not path:
        return
    try:
        threading.Thread(
            target=lambda: sa.WaveObject.from_wave_file(path).play().wait_done(),
            daemon=True
        ).start()
    except Exception as e:
        logger.warning(f"Sound play failed {path}: {e}")

def int16_bytes_to_float32_mono(buf: bytes, in_rate: int, out_rate: int = 16000) -> np.ndarray:
    if not buf:
        return np.array([], dtype=np.float32)
    try:
        if in_rate != out_rate:
            buf, _ = audioop.ratecv(buf, 2, 1, in_rate, out_rate, None)
        return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return np.array([], dtype=np.float32)

# ============================================================
# ----------------------- AUDIO STREAM -----------------------
# ============================================================

class AudioStream:
    """
    Double-buffered mic capture:
    - 0.5s quiet flips buffer
    - 2.5s continuous quiet stops recording
    - hard switch at 70s
    """

    def __init__(self, rate=DEFAULT_RATE, channels=DEFAULT_CHANNELS,
                 chunk=DEFAULT_CHUNK, silence_threshold=DEFAULT_SILENCE_THRESHOLD,
                 silence_duration=DEFAULT_SILENCE_DURATION,
                 buffer_switch_silence=DEFAULT_BUFFER_SWITCH_SILENCE,
                 max_recording_time=DEFAULT_MAX_RECORDING_TIME,
                 hard_switch_time=DEFAULT_HARD_SWITCH_TIME,
                 start_sound=DEFAULT_START_SOUND, stop_sound=DEFAULT_STOP_SOUND,
                 device_index=DEFAULT_DEVICE_INDEX):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.buffer_switch_silence = buffer_switch_silence
        self.max_recording_time = max_recording_time
        self.hard_switch_time = hard_switch_time
        self.start_sound = start_sound
        self.stop_sound = stop_sound
        self.device_index = device_index
        self.bytes_per_sample = 2

        self.buffers = [bytearray(), bytearray()]
        self.active = 0
        self.buffer_counter = 0

        self.lock = threading.Lock()
        self.chunk_queue = Queue()
        self.running = False
        self.p = None
        self.stream = None
        self.silence_time = 0.0
        self.has_speech = False
        self.start_time = 0.0
        self.last_switch_time = 0.0

    def start(self):
        if self.running:
            return
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.device_index,
                stream_callback=self._callback
            )
        except Exception as e:
            if self.p:
                self.p.terminate()
            raise RuntimeError(f"Failed to open input device: {e}")

        with self.lock:
            for b in self.buffers:
                b.clear()
            self.active = 0
            self.buffer_counter = 0
            self.silence_time = 0.0
            self.has_speech = False
            while not self.chunk_queue.empty():
                try:
                    self.chunk_queue.get_nowait()
                except Empty:
                    break

        self.start_time = time.time()
        self.last_switch_time = self.start_time
        self.running = True
        self.stream.start_stream()
        time.sleep(0.15)
        play_sound(self.start_sound)
        logger.info("Recording started (switch=%.1fs, stop=%.1fs, hard=%.1fs)",
                    self.buffer_switch_silence, self.silence_duration, self.hard_switch_time)

    def stop(self):
        if not self.running:
            return
        self.running = False
        with self.lock:
            if len(self.buffers[self.active]) > 0:
                dur = len(self.buffers[self.active]) / (self.rate * self.bytes_per_sample)
                self.chunk_queue.put(AudioChunk(
                    data=bytes(self.buffers[self.active]),
                    buffer_id=self.buffer_counter,
                    timestamp=time.time(),
                    duration=dur
                ))
                self.buffers[self.active].clear()
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception:
            pass
        play_sound(self.stop_sound)
        logger.info("Recording stopped (%.2fs)", time.time() - self.start_time)

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            with self.lock:
                self.buffers[self.active].extend(in_data)
                rms = audioop.rms(in_data, 2)
                quiet = rms < self.silence_threshold
                dur_chunk = frame_count / float(self.rate)

                if quiet and self.has_speech:
                    self.silence_time += dur_chunk
                else:
                    if not quiet:
                        self.has_speech = True
                    self.silence_time = 0.0

                # Switch buffer if 0.5s quiet after speech
                time_since_switch = time.time() - self.last_switch_time
                if self.has_speech and self.silence_time >= self.buffer_switch_silence:
                    data_len = len(self.buffers[self.active])
                    if data_len > int(self.rate * 0.2) * self.bytes_per_sample:  # >0.2s
                        dur = data_len / (self.rate * self.bytes_per_sample)
                        self.chunk_queue.put(AudioChunk(
                            data=bytes(self.buffers[self.active]),
                            buffer_id=self.buffer_counter,
                            timestamp=time.time(),
                            duration=dur
                        ))
                        logger.debug(f"Buffer {self.buffer_counter} pushed ({dur:.2f}s, rms={rms})")
                    self.active = 1 - self.active
                    self.buffers[self.active].clear()
                    self.buffer_counter += 1
                    self.silence_time = 0.0
                    self.has_speech = False
                    self.last_switch_time = time.time()

                # Hard switch if >70s
                elif time_since_switch >= self.hard_switch_time:
                    dur = len(self.buffers[self.active]) / (self.rate * self.bytes_per_sample)
                    self.chunk_queue.put(AudioChunk(
                        data=bytes(self.buffers[self.active]),
                        buffer_id=self.buffer_counter,
                        timestamp=time.time(),
                        duration=dur
                    ))
                    self.active = 1 - self.active
                    self.buffers[self.active].clear()
                    self.buffer_counter += 1
                    self.silence_time = 0.0
                    self.has_speech = False
                    self.last_switch_time = time.time()
                    logger.warning("Hard switch triggered at %.1fs", time_since_switch)
        except Exception as e:
            logger.error(f"Callback error: {e}")
            traceback.print_exc()
        return (None, pyaudio.paContinue)

    def get_next_chunk(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        try:
            return self.chunk_queue.get(timeout=timeout)
        except Empty:
            return None

    def should_stop(self) -> bool:
        with self.lock:
            elapsed = time.time() - self.start_time
            timeout = elapsed >= self.max_recording_time
            long_silence = self.has_speech and self.silence_time >= self.silence_duration
            if timeout:
                logger.info("Max recording time reached")
            elif long_silence:
                logger.info("Silence detected (%.2fs)", self.silence_time)
            return timeout or long_silence

# ============================================================
# -------------------- TRANSCRIBER + FIXER -------------------
# ============================================================

class WhisperTranscriber:
    def __init__(self, model_name=DEFAULT_WHISPER_MODEL):
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_f32: np.ndarray, language='en') -> str:
        if len(audio_f32) == 0:
            return ""
        try:
            result = self.model.transcribe(audio_f32, language=language)
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return ""

class GemmaFixer:
    def __init__(self, model_name=DEFAULT_GEMMA_MODEL,
                 system_message=SYSTEM_PROMPT_2_AUDIO,
                 temperature=DEFAULT_MODEL_TEMPERATURE,
                 max_tokens=DEFAULT_MODEL_MAX_TOKENS):
        self.client = GemmaClient()
        self.bot = GemmaConversationBot(
            client=self.client,
            system_message=system_message,
            k=1,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def fix(self, text: str) -> str:
        if not text:
            return ""
        try:
            return self.bot.UILLM(messegeIn=f"fix the following text:\n{text}").strip()
        except Exception as e:
            logger.error(f"Gemma fixer failed: {e}")
            return text.strip()

# ============================================================
# ------------------------- SERVICE --------------------------
# ============================================================

class STTService:
    def __init__(self, audio: AudioStream, transcriber: WhisperTranscriber, fixer: GemmaFixer):
        self.audio = audio
        self.transcriber = transcriber
        self.fixer = fixer

    def _process_chunk(self, chunk: AudioChunk, language: str) -> str:
        if chunk.duration < 0.25:
            return ""
        audio_f32 = int16_bytes_to_float32_mono(chunk.data, self.audio.rate, 16000)
        text = self.transcriber.transcribe(audio_f32, language)
        if not text:
            return ""
        fixed = self.fixer.fix(text)
        logger.info(f"[CHUNK {chunk.buffer_id}] {fixed}")
        return fixed

    def listen_until_silence(self, language='en') -> str:
        texts = []
        self.audio.start()
        try:
            while True:
                if self.audio.should_stop():
                    self.audio.stop()
                chunk = self.audio.get_next_chunk(timeout=0.5)
                if chunk:
                    result = self._process_chunk(chunk, language)
                    if result:
                        texts.append(result)
                if not self.audio.running and self.audio.chunk_queue.empty():
                    break
        finally:
            self.audio.stop()
        final_text = " ".join(texts).strip()
        logger.info(f"[FINAL TRANSCRIPTION] {final_text}")
        return final_text

# ============================================================
# -------------------------- FLASK ---------------------------
# ============================================================

def create_app(service: STTService):
    app = Flask(__name__)

    @app.route("/SpeechToTextService/Listen", methods=["POST"])
    def listen():
        lang = (request.args.get("lang") or "en").strip()
        txt = service.listen_until_silence(language=lang)
        return jsonify({"transcription": txt})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app

# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX)
    p.add_argument("--rate", type=int, default=DEFAULT_RATE)
    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--silence-threshold", type=int, default=DEFAULT_SILENCE_THRESHOLD)
    p.add_argument("--silence-duration", type=float, default=DEFAULT_SILENCE_DURATION)
    p.add_argument("--buffer-switch-silence", type=float, default=DEFAULT_BUFFER_SWITCH_SILENCE)
    p.add_argument("--max-recording-time", type=float, default=DEFAULT_MAX_RECORDING_TIME)
    p.add_argument("--hard-switch-time", type=float, default=DEFAULT_HARD_SWITCH_TIME)
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    p.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()

    if a.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    audio = AudioStream(rate=a.rate, channels=a.channels, chunk=a.chunk,
                        silence_threshold=a.silence_threshold,
                        silence_duration=a.silence_duration,
                        buffer_switch_silence=a.buffer_switch_silence,
                        max_recording_time=a.max_recording_time,
                        hard_switch_time=a.hard_switch_time,
                        device_index=a.device_index)
    trans = WhisperTranscriber(a.whisper_model)
    fixer = GemmaFixer(a.gemma_model)
    service = STTService(audio, trans, fixer)
    app = create_app(service)
    logger.info(f"Server running on {a.host}:{a.port}")
    from waitress import serve
    app.run(host=a.host, port=a.port, debug=a.debug, use_reloader=False)

if __name__ == "__main__":
    main()




# python sttooc.py --host 127.0.0.1 --port 6003 --device-index 0 --rate 44100 --channels 1 --chunk 1024 --silence-threshold 500 --silence-duration 2.5 --buffer-switch-silence 0.5 --max-recording-time 60 --hard-switch-time 70 --whisper-model large-v3-turbo --gemma-model gemma-2-2b-it-GGUF --debug

# curl -X POST "http://127.0.0.1:6003/SpeechToTextService/Listen?lang=en"







'''
python sttooc.py --host 127.0.0.1 --port 6003 --device-index 0 --rate 44100 --channels 1 --chunk 1024 --min-window 60 --max-window 70 --silence-threshold 250 --silence-duration 2.5 --buffer-switch-silence 0.5 --whisper-model large-v3-turbo --gemma-model gemma-2-2b-it-GGUF --debug


| Option                             | Description                                                           |
| :--------------------------------- | :-------------------------------------------------------------------- |
| `--host 127.0.0.1`                 | Bind Flask server to localhost (accessible only from your PC).        |
| `--port 6003`                      | Server port (change if you have another service using this port).     |
| `--device-index 0`                 | Audio input device (see `/devices` endpoint to check indices).        |
| `--rate 44100`                     | Sample rate for microphone capture.                                   |
| `--channels 1`                     | Record mono input (saves bandwidth and model load).                   |
| `--chunk 1024`                     | PyAudio frame size per read — smaller = lower latency.                |
| `--min-window 60`                  | Record at least 60 seconds before checking for silence.               |
| `--max-window 70`                  | Force a buffer switch if you reach 70 seconds without silence.        |
| `--silence-threshold 250`          | RMS level below which audio is considered "quiet".                    |
| `--silence-duration 2.5`           | Continuous quiet time (sec) to stop recording.                        |
| `--whisper-model large-v3-turbo`   | Model for speech recognition (`meta` for Hebrew→English translation). |
| `--gemma-model gemma-2-2b-it-GGUF` | Local Gemma model for text correction.                                |
| `--debug`                          | Enables verbose logs for debugging and development.                   |




curl -X POST "http://127.0.0.1:6003/SpeechToTextService/Listen?lang=en&fix=true&min_window=60&max_window=70&silence_duration=2.5&silence_threshold=250&device_index=0&whisper_model=large-v3-turbo&gemma_model=gemma-2-2b-it-GGUF"

'''