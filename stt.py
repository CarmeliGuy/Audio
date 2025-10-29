# https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
# pipline parameters

# ( model: PreTrainedModelfeature_extractor: typing.Union[ForwardRef('SequenceFeatureExtractor'), str, NoneType] = Nonetokenizer: typing.Optional[transformers.tokenization_utils.PreTrainedTokenizer] = Nonedecoder: typing.Union[ForwardRef('BeamSearchDecoderCTC'), str, NoneType] = Nonedevice: typing.Union[int, ForwardRef('torch.device'), NoneType] = None**kwargs )

# =====================================================================================
# =====================================================================================
# =====================================================================================
import os
# 🚨 These must come before importing datasets or torch
os.environ["HF_DATASETS_AUDIO_DISABLE_TORCHCODEC"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# from AudioModels.playSound import PlaySound     #PlaySound(audio, sample_rate)
import torch
import soundfile as sf               # ✅ you forgot to import this
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
model_id = r"D:\AcceleratorProj\AudioModels\whisper-large-v3-turbo"
audio_path = r"D:\AcceleratorProj\audio detect red car.wav"

# ----------------------------------------------------------
# DEVICE + MODEL
# ----------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    device_map="auto",  # ✅ let Accelerate choose the GPU automatically
)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
)

# ----------------------------------------------------------
# LOAD AND RUN LOCAL AUDIO
# ----------------------------------------------------------
# audio, sample_rate = sf.read(audio_path)
import librosa
audio_path = r"D:\AcceleratorProj\audio detect red car.wav"
audio, sample_rate = librosa.load(audio_path, sr=16000)  # resample to 16 kHz (Whisper prefers this)

print(f"Loaded audio: {audio.shape} samples @ {sample_rate} Hz")

PlaySound(audio, sample_rate)
import time
for i in range(5):
    print(f"[Main] still running... {i}")
    time.sleep(1)

print("[Main] done.")


# Run inference
# result = pipe({"array": audio, "sampling_rate": sample_rate})
result = pipe(
    {"array": audio, "sampling_rate": sample_rate},
    generate_kwargs={"task": "transcribe", "language": "en"}  # or "he" for Hebrew
)
# ----------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------
print("\n=== Transcription Result ===")
print(result["text"])

# =====================================================================================
# =====================================================================================
# =====================================================================================






# task (str) — The task defining which pipeline will be returned. Currently accepted tasks are:
# "audio-classification": will return a AudioClassificationPipeline.
# "automatic-speech-recognition": will return a AutomaticSpeechRecognitionPipeline.
# "depth-estimation": will return a DepthEstimationPipeline.
# "document-question-answering": will return a DocumentQuestionAnsweringPipeline.
# "feature-extraction": will return a FeatureExtractionPipeline.
# "fill-mask": will return a FillMaskPipeline:.
# "image-classification": will return a ImageClassificationPipeline.
# "image-feature-extraction": will return an ImageFeatureExtractionPipeline.
# "image-segmentation": will return a ImageSegmentationPipeline.
# "image-text-to-text": will return a ImageTextToTextPipeline.
# "image-to-image": will return a ImageToImagePipeline.
# "image-to-text": will return a ImageToTextPipeline.
# "keypoint-matching": will return a KeypointMatchingPipeline.
# "mask-generation": will return a MaskGenerationPipeline.
# "object-detection": will return a ObjectDetectionPipeline.
# "question-answering": will return a QuestionAnsweringPipeline.
# "summarization": will return a SummarizationPipeline.
# "table-question-answering": will return a TableQuestionAnsweringPipeline.
# "text2text-generation": will return a Text2TextGenerationPipeline.
# "text-classification" (alias "sentiment-analysis" available): will return a TextClassificationPipeline.
# "text-generation": will return a TextGenerationPipeline:.
# "text-to-audio" (alias "text-to-speech" available): will return a TextToAudioPipeline:.
# "token-classification" (alias "ner" available): will return a TokenClassificationPipeline.
# "translation": will return a TranslationPipeline.
# "translation_xx_to_yy": will return a TranslationPipeline.
# "video-classification": will return a VideoClassificationPipeline.
# "visual-question-answering": will return a VisualQuestionAnsweringPipeline.
# "zero-shot-classification": will return a ZeroShotClassificationPipeline.
# "zero-shot-image-classification": will return a ZeroShotImageClassificationPipeline.
# "zero-shot-audio-classification": will return a ZeroShotAudioClassificationPipeline.
# "zero-shot-object-detection": will return a ZeroShotObjectDetectionPipeline.


# https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline




import librosa
import sounddevice as sd
import numpy as np
from multiprocessing import Process, shared_memory, Event
import time

def _child_play(name, shape, dtype, sr, ready_event: Event):
    """
    Child process attaches to shared memory and plays the audio.
    Signals the parent when ready.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        audio = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Notify parent we successfully attached
        ready_event.set()

        print(f"[Child] Attached to shared memory: {name}")
        sd.play(audio, samplerate=sr)
        sd.wait()

        shm.close()
        print("[Child] Playback finished.")
    except Exception as e:
        print(f"[Child] ERROR: {e}")
        ready_event.set()  # still signal to avoid deadlock


def PlaySound(audio: np.ndarray, sr: int):
    """
    Plays a NumPy audio buffer in a separate process using shared memory.
    Automatically waits until the child attaches safely.
    """
    shm = shared_memory.SharedMemory(create=True, size=audio.nbytes)
    np.ndarray(audio.shape, dtype=audio.dtype, buffer=shm.buf)[:] = audio

    ready_event = Event()
    p = Process(target=_child_play, args=(shm.name, audio.shape, audio.dtype, sr, ready_event))
    p.start()

    # Wait for child process to confirm memory attachment
    if not ready_event.wait(timeout=3.0):
        print("[Parent] WARNING: Child did not attach within timeout.")
    else:
        print("[Parent] Child attached successfully.")

    shm.close()  # now safe to close handle in parent
    print(f"[Parent] Playback running in process PID={p.pid}")