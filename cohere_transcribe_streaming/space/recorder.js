const recorder = element.querySelector(".cohere-recorder");
const language = recorder.querySelector('[data-role="language"]');
const recordButton = recorder.querySelector('[data-action="record"]');
const stopButton = recorder.querySelector('[data-action="stop"]');
const clearButton = recorder.querySelector('[data-action="clear"]');
const status = recorder.querySelector('[data-role="status"]');
const transcript = recorder.querySelector('[data-role="transcript"]');
const badge = recorder.querySelector('[data-role="badge"]');
const time = recorder.querySelector('[data-role="time"]');
const progress = recorder.querySelector('[data-role="progress"]');
const maximumSeconds = 25;
let currentRecording = null;

const workletCode = `
class MicrophoneRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(Math.round(sampleRate * 0.4));
    this.bufferLength = 0;
    this.totalSamples = 0;
    this.stopped = false;
    this.port.onmessage = () => this.finish();
  }
  sendAudio() {
    if (!this.bufferLength) return;
    const samples = this.buffer.slice(0, this.bufferLength);
    this.port.postMessage({ type: "audio", samples }, [samples.buffer]);
    this.bufferLength = 0;
  }
  finish() {
    if (this.stopped) return;
    this.stopped = true;
    this.sendAudio();
    this.port.postMessage({ type: "end" });
  }
  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels.length) return true;
    for (let index = 0; index < channels[0].length; index++) {
      let sample = 0;
      for (const channel of channels) sample += channel[index];
      this.buffer[this.bufferLength++] = sample / channels.length;
      this.totalSamples++;
      if (this.bufferLength === this.buffer.length) this.sendAudio();
      if (this.totalSamples >= sampleRate * ${maximumSeconds}) {
        this.finish();
        return false;
      }
    }
    return true;
  }
}
registerProcessor("cohere-microphone", MicrophoneRecorder);
`;

function setPhase(phase, message) {
  recorder.dataset.phase = phase;
  status.textContent = message;
  recordButton.disabled = ["opening", "recording", "finishing"].includes(phase);
  stopButton.disabled = phase !== "recording";
  language.disabled = recordButton.disabled;
}

function showDuration(seconds) {
  const duration = Math.min(maximumSeconds, seconds);
  time.textContent = `${duration.toFixed(1)} / ${maximumSeconds} seconds`;
  progress.value = duration;
}

function stopMicrophone(recording) {
  clearTimeout(recording.timer);
  recording.stream?.getTracks().forEach(track => track.stop());
  recording.source?.disconnect();
  recording.worklet?.disconnect();
  if (recording.context && recording.context.state !== "closed") {
    recording.context.close().catch(() => {});
  }
}

function finishInBackground(recording) {
  recording.uploads.catch(() => {}).then(() => {
    if (recording.id) return server.finish_recording({ session_id: recording.id });
  }).catch(() => {});
}

function failRecording(recording, error) {
  if (!recording || recording.canceled) return;
  recording.canceled = true;
  clearTimeout(recording.watchdog);
  stopMicrophone(recording);
  finishInBackground(recording);
  if (currentRecording !== recording) return;
  const message = error?.name === "NotAllowedError"
    ? "Microphone access was denied. Allow access in your browser and try again."
    : error?.message || "The recording could not be transcribed. Please try again.";
  setPhase("error", message);
}

function sendAudio(recording, samples) {
  recording.samples += samples.length;
  if (currentRecording === recording) {
    showDuration(recording.samples / recording.context.sampleRate);
  }
  recording.uploads = recording.uploads.then(async () => {
    if (recording.canceled) return;
    const result = await server.receive_audio({ session_id: recording.id, samples: Array.from(samples) });
    if (result.full) stopRecording(recording).catch(error => failRecording(recording, error));
  });
  recording.uploads.catch(error => failRecording(recording, error));
}

async function stopRecording(recording = currentRecording) {
  if (!recording || recording.canceled) return;
  if (recording.stopping) return recording.stopping;
  if (currentRecording === recording) setPhase("finishing", "Finishing the transcript…");
  recording.stopping = (async () => {
    let timeout;
    try {
      recording.worklet.port.postMessage({ type: "stop" });
      await Promise.race([
        recording.ended,
        new Promise((resolve, reject) => {
          timeout = setTimeout(() => reject(new Error("The microphone stopped responding. Please try again.")), 2000);
        }),
      ]);
    } finally {
      clearTimeout(timeout);
      stopMicrophone(recording);
    }
    await recording.uploads;
    if (!recording.canceled) await server.finish_recording({ session_id: recording.id });
  })();
  return recording.stopping;
}

async function showTranscript(recording, update) {
  if (!recording || recording.canceled || update?.session_id !== recording.id) return;
  transcript.textContent = update.text || "";
  if (update.error) throw new Error(update.error);
  status.textContent = recording.stopping ? "Finishing the transcript…" : update.status || "Listening…";
  if (!update.final) return;
  clearTimeout(recording.watchdog);
  stopMicrophone(recording);
  if (recording.stopping) await recording.stopping;
  if (currentRecording === recording && !recording.canceled) {
    badge.textContent = "Final";
    setPhase("done", update.status || "Final transcript");
  }
}

async function startRecording() {
  const recording = {
    id: null,
    canceled: false,
    samples: 0,
    uploads: Promise.resolve(),
  };
  currentRecording = recording;
  transcript.textContent = "";
  badge.textContent = "Live";
  showDuration(0);
  setPhase("opening", "Opening the microphone…");
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error("Microphone streaming needs a recent browser and a secure connection.");
    }
    try {
      recording.context = new AudioContext({ sampleRate: 16000 });
    } catch {
      recording.context = new AudioContext();
    }
    await recording.context.resume();
    if (recording.canceled) return stopMicrophone(recording);
    recording.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    if (recording.canceled) return stopMicrophone(recording);
    const workletUrl = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
    try {
      await recording.context.audioWorklet.addModule(workletUrl);
    } finally {
      URL.revokeObjectURL(workletUrl);
    }
    if (recording.canceled) return stopMicrophone(recording);
    recording.id = await server.begin_recording({ language: language.value, sample_rate: recording.context.sampleRate });
    if (recording.canceled) {
      stopMicrophone(recording);
      return finishInBackground(recording);
    }
    recording.ended = new Promise(resolve => { recording.resolveEnd = resolve; });
    recording.worklet = new AudioWorkletNode(recording.context, "cohere-microphone");
    recording.worklet.port.onmessage = ({ data }) => {
      if (data.type === "audio" && !recording.canceled) sendAudio(recording, data.samples);
      if (data.type === "end") {
        recording.resolveEnd();
        stopRecording(recording).catch(error => failRecording(recording, error));
      }
    };
    recording.source = recording.context.createMediaStreamSource(recording.stream);
    recording.source.connect(recording.worklet);
    recording.worklet.connect(recording.context.destination);
    recording.timer = setTimeout(() => {
      stopRecording(recording).catch(error => failRecording(recording, error));
    }, maximumSeconds * 1000);
    recording.watchdog = setTimeout(() => {
      failRecording(recording, new Error("Transcription is taking too long. A GPU may be unavailable; please try again."));
    }, 180000);
    setPhase("recording", "Recording · waiting for the first words…");
    trigger("click", { session_id: recording.id });
  } catch (error) {
    failRecording(recording, error);
  }
}

function clearRecording() {
  const recording = currentRecording;
  currentRecording = null;
  if (recording) {
    recording.canceled = true;
    clearTimeout(recording.watchdog);
    stopMicrophone(recording);
    finishInBackground(recording);
  }
  transcript.textContent = "";
  badge.textContent = "Live";
  showDuration(0);
  setPhase("idle", "Ready when you are.");
}

recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", () => {
  const recording = currentRecording;
  stopRecording(recording).catch(error => failRecording(recording, error));
});
clearButton.addEventListener("click", clearRecording);
window.addEventListener("pagehide", clearRecording);
watch("value", () => {
  const recording = currentRecording;
  showTranscript(recording, props.value).catch(error => failRecording(recording, error));
});
