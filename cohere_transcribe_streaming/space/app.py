"""Record microphone audio and display a streaming Cohere transcript."""

import spaces

import copy
import json
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soxr
import torch

APP_DIRECTORY = Path(__file__).resolve().parent
if not (APP_DIRECTORY / "model.py").exists():
    sys.path.insert(0, str(APP_DIRECTORY.parent))

from model import CohereStreaming, SR


MAX_SECONDS = 25
CHUNK_SAMPLES = 1280
SESSION_DIRECTORY = Path(tempfile.mkdtemp(prefix="cohere-recordings-"))
UPLOAD_LOCK = threading.Lock()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
recognizer = CohereStreaming(device=DEVICE, dtype=DTYPE)


def recording_folder(session_id: str, must_exist: bool = True) -> Path:
    try:
        valid_id = str(uuid.UUID(session_id))
    except (ValueError, TypeError, AttributeError):
        raise gr.Error("Invalid recording session.") from None
    if valid_id != session_id:
        raise gr.Error("Invalid recording session.")
    folder = SESSION_DIRECTORY / session_id
    if must_exist and not folder.is_dir():
        raise gr.Error("This recording has ended. Start a new recording.")
    return folder


def write_metadata(folder: Path, metadata: dict) -> None:
    temporary_file = folder / "metadata.tmp"
    temporary_file.write_text(json.dumps(metadata))
    temporary_file.replace(folder / "metadata.json")


def remove_expired_recordings() -> None:
    for folder in SESSION_DIRECTORY.iterdir():
        try:
            expired = time.time() - folder.stat().st_mtime > 600
        except FileNotFoundError:
            continue
        if expired:
            shutil.rmtree(folder, ignore_errors=True)


def begin_recording(data: dict) -> str:
    """Create an isolated recording before the browser starts uploading audio."""
    language = data["language"]
    sample_rate = data["sample_rate"]
    recognizer.processor.get_decoder_prompt_ids(language=language)
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise gr.Error("The microphone sample rate must be between 8 and 96 kHz.")
    remove_expired_recordings()
    session_id = str(uuid.uuid4())
    folder = SESSION_DIRECTORY / session_id
    folder.mkdir()
    write_metadata(folder, {"language": language, "sample_rate": sample_rate, "samples": 0, "chunks": 0})
    return session_id


def receive_audio(data: dict) -> dict:
    """Store one microphone chunk, keeping its original order and sample rate."""
    session_id = data["session_id"]
    samples = data["samples"]
    folder = recording_folder(session_id)
    with UPLOAD_LOCK:
        metadata = json.loads((folder / "metadata.json").read_text())
        sample_rate = metadata["sample_rate"]
        if len(samples) > 2 * sample_rate:
            raise gr.Error("Send audio in chunks of at most two seconds.")
        if (folder / "finished").exists():
            return {"seconds": metadata["samples"] / sample_rate, "full": True}
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise gr.Error("Microphone audio must contain finite mono samples.")
        available_samples = MAX_SECONDS * sample_rate - metadata["samples"]
        audio = np.clip(audio[:available_samples], -1.0, 1.0)
        if audio.size:
            chunk_file = folder / f"{metadata['chunks']:06d}.npy"
            temporary_file = chunk_file.with_suffix(".tmp")
            with temporary_file.open("wb") as output:
                np.save(output, audio, allow_pickle=False)
            temporary_file.replace(chunk_file)
            metadata["chunks"] += 1
            metadata["samples"] += audio.size
            write_metadata(folder, metadata)
        full = metadata["samples"] >= MAX_SECONDS * sample_rate
        if full:
            (folder / "finished").touch()
        return {"seconds": metadata["samples"] / sample_rate, "full": full}


def finish_recording(data: dict) -> dict:
    """Mark the recording complete after every microphone chunk has arrived."""
    folder = recording_folder(data["session_id"], must_exist=False)
    with UPLOAD_LOCK:
        if folder.is_dir():
            (folder / "finished").touch()
    return {"ok": True}


def transcript_update(session_id: str, stream: CohereStreaming, text: str, status: str, final: bool = False) -> dict:
    return {
        "session_id": session_id,
        "text": text,
        "status": status,
        "seconds": round(stream._n_samples / SR, 2),
        "final": final,
    }


@spaces.GPU(duration=45)
def transcribe(session_id: str):
    """Keep the GPU for one recording and yield text as new audio arrives."""
    folder = recording_folder(session_id)
    try:
        metadata = json.loads((folder / "metadata.json").read_text())
        stream = copy.copy(recognizer)
        stream.featurizer = copy.deepcopy(recognizer.featurizer)
        stream.reset(language=metadata["language"])
        sample_rate = metadata["sample_rate"]
        resampler = soxr.ResampleStream(sample_rate, SR, 1, dtype="float32") if sample_rate != SR else None
        pending_audio = np.empty(0, dtype=np.float32)
        chunk_index = 0
        text = ""
        audio_complete = False
        deadline = time.monotonic() + 35
        yield transcript_update(session_id, stream, text, "Listening…")

        while time.monotonic() < deadline:
            chunk_file = folder / f"{chunk_index:06d}.npy"
            if not chunk_file.exists():
                if (folder / "finished").exists():
                    audio_complete = True
                    break
                time.sleep(0.03)
                continue
            audio = np.load(chunk_file, allow_pickle=False)
            chunk_file.unlink()
            chunk_index += 1
            if resampler is not None:
                audio = resampler.resample_chunk(audio)
            pending_audio = np.concatenate([pending_audio, audio])
            while pending_audio.size >= CHUNK_SAMPLES:
                text = stream.step(pending_audio[:CHUNK_SAMPLES])
                pending_audio = pending_audio[CHUNK_SAMPLES:]
            yield transcript_update(session_id, stream, text, "Transcribing…")

        if not audio_complete:
            raise gr.Error("Recording took too long to process. Please record a shorter clip and try again.")
        yield transcript_update(session_id, stream, text, "Finishing…")
        if resampler is not None:
            tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
            pending_audio = np.concatenate([pending_audio, tail])
        if pending_audio.size:
            stream.step(pending_audio)
        text = stream.finish()
        yield transcript_update(session_id, stream, text, "Final transcript", final=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def transcribe_from_recorder(event: gr.EventData):
    folder = recording_folder(event.session_id, must_exist=False)
    try:
        yield from transcribe(event.session_id)
    except Exception as error:
        yield {
            "session_id": event.session_id,
            "text": "",
            "error": str(error),
            "status": "Transcription failed",
            "final": True,
        }
    finally:
        shutil.rmtree(folder, ignore_errors=True)


with gr.Blocks(title="Streaming Cohere Transcribe") as demo:
    recorder = gr.HTML(
        html_template=(APP_DIRECTORY / "recorder.html").read_text(),
        js_on_load=(APP_DIRECTORY / "recorder.js").read_text(),
        server_functions=[begin_recording, receive_audio, finish_recording],
    )
    recorder.click(
        transcribe_from_recorder,
        inputs=None,
        outputs=recorder,
        concurrency_id="cohere-transcription",
        concurrency_limit=1,
        trigger_mode="multiple",
        show_progress="hidden",
        api_visibility="private",
    )
    session = gr.Textbox(visible=False)
    output = gr.JSON(visible=False)
    gr.Button(visible=False).click(
        transcribe,
        inputs=session,
        outputs=output,
        api_name="transcribe",
        concurrency_id="cohere-transcription",
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue().launch()
