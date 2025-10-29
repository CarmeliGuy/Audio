# STTS.py

# speech to text server




import threading
import time
import pyaudio
import wave
import audioop
import whisper
import simpleaudio as sa
from flask import Flask, jsonify
from clientGemma.clientLLM import GemmaConversationBot,GemmaClient
from clientGemma import clientLLM
app = Flask(__name__)
# 10.87.226.165
# ================== Audio Recording Settings ==================
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FILE_PATH = "./recorded_audio.wav"


# Sound file paths (WAV format recommended)
START_SOUND = r"./sounds/start.wav"
STOP_SOUND = r"./sounds/stop.wav"



# ================== prompt for fixing the transcribed text  ==================
systemprompt2audio = ''' You are a transcription-refinement assistant.
Your input is a single raw transcript sentence. Your task is to output **only**the fully corrected sentence, with any mis-transcribed or ambiguous words replaced by the most contextually appropriate terms.
Do not add any commentary, explanations, brackets, or questions - just the clean sentence.

steps:
1.Scan the sentence for any words that could be wrong(homophones, mis-hearings, garbled tokens.)
2.use the entire sentence context and general domain knowledge to pick tge single best replacement for each.
3.If more than one word seems plausible,choose the one that makes the sentence most natural and coherent.
4.Consider the words might have changed in example:bin->been,trail->tail, clay->claim, cite->site, see->sea, could->would based on the context.
5.Preserve the speaker's original style and phrasing where ever possible.
6.Output exactly one line:the corrected sentence. 
7.Keep the sentence structure the same.

Begin.


'''

# Shared state
is_recording = False
new_audio = False

# # Load Whisper model
# model = whisper.load_model("base")
# print("**Whisper model loaded**")
model_id = "large-v3-turbo"
local_model_path = r"D:\AcceleratorProj\AudioModels\whisper-large-v3-turbo"
model = whisper.load_model(model_id)
print(f"**Whisper model loaded from: {local_model_path}**")

# ================== function to play the sounds at the start and at the end of the recording ==================
def play_sound(file_path):
    """
    Play a WAV file using simpleaudio (cross-platform and reliable).
    """
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")



# ================== A function that record the input voice in to a wav file ==================
def record_audio_with_silence_detection():
    """
    Start recording audio and play start sound in parallel.
    Stop automatically after 2 seconds of silence.
    """
    global is_recording, new_audio
    is_recording = True
    new_audio = False
    audio_frames = []



    # Setup audio input
    p = pyaudio.PyAudio()
    input_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    output_stream = p.open(format=FORMAT, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
    # Play start sound asynchronously
    threading.Thread(target=play_sound, args=(START_SOUND,), daemon=True).start()
    print("Recording started...")
    silence_threshold = 500
    silence_duration = 2
    silent_time = 0

    try:
        while is_recording:
            data = input_stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(data)

            rms = audioop.rms(data, 2)
            if rms < silence_threshold:
                silent_time += CHUNK / RATE
            else:
                silent_time = 0

            if silent_time >= silence_duration:
                print("Silence detected, stopping recording...")
                play_sound(STOP_SOUND)
                break

    finally:
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()
        print("Recording stopped.")

        # Save to WAV
        if audio_frames:
            with wave.open(FILE_PATH, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_frames))
            print(f"Audio saved to {FILE_PATH}")
            new_audio = True

        is_recording = False







# ================== A function that activate the model to transcribe the wav file  ==================

def transcribe_audio(audio_file):
    """
    Transcribe an audio file using the Whisper model.
    """
    print("Transcribing audio...")
    try:
        result = model.transcribe(audio_file, language='en')
        print(result)  # Optional: inspect full result
        text = result["text"].strip()
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return None



# ================== The function used to send the transcribed text to be fixed by the LLM (currently: Gemma-3-4B In LMstudio ==================

def fix_text_with_llm(raw_text):
    client = GemmaClient()
    conversation_bot = GemmaConversationBot(
        client=client,
        system_message=systemprompt2audio,
        k=100,  # Keep last 20 messages in memory
        model="gemma-2-2b-it-GGUF",
        temperature=0.7,
        max_tokens=4096
    )

    user_text = conversation_bot.UILLM(messegeIn=f"fix the following text:\n \n {raw_text}")

    return user_text


# ================== The whisper server endpoint, implementing the work flow: **Save user recording** --> **Model transcribe speech 2 text ** --> **LLM fixing**  ==================


@app.route("/SpeechToTextService/Listen", methods=["POST"])
def listen_and_transcribe():
    global is_recording, new_audio

    if is_recording:
        return jsonify({"error": "Recording already in progress"}), 400

    threading.Thread(target=record_audio_with_silence_detection, daemon=True).start()

    while is_recording:
        time.sleep(0.1)

    if new_audio:
        text = transcribe_audio(FILE_PATH)
        print(f"transcribed:{text}")
        text = fix_text_with_llm(text)
        print(f"transcribed:{text}")

        if text:

            return jsonify({"transcription": text}), 200
        return jsonify({"error": "Transcription failed"}), 500

    return jsonify({"error": "No audio recorded"}), 400

if __name__ == "__main__":
    ip = input("Enter IP (default 127.0.0.1): ") or "127.0.0.1"
    port_input = input("Enter port (default 6003): ") or "6003"

    try:
        port = int(port_input)
    except ValueError:
        print("Invalid port. Using default 6003.")
        port = 6003

    print(f"Server running on {ip}:{port}")
    app.run(host=ip, port=port, use_reloader= False, debug=True)

