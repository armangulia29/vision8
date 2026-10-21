# Blind Assist hybrid prototype

## Install on Windows

In PowerShell, from this folder:

```powershell
py -3.14 -m pip install --upgrade pip
py -3.14 -m pip install ultralytics opencv-python numpy pyttsx3 sounddevice onnxruntime
```

The script continues if optional sound, speech, OCR, depth, or cloud packages fail. ONNX Runtime currently publishes Windows x64 wheels for CPython 3.14; if a package install on your exact interpreter fails, the audio branch stays optional and can be disabled with `--no-audio`. Install OCR separately with `py -3.14 -m pip install easyocr`; EasyOCR depends on PyTorch/torchvision and may be the first dependency to fail on a new Python 3.14 setup. Use Python 3.12 for the easiest OCR fallback. `pyttsx3`'s published Python classifiers currently stop at 3.13, so the code falls back to Windows System.Speech; its speech path is optional. PyTorch currently documents Windows support for Python 3.10–3.14; if `ultralytics`/PyTorch dependency resolution fails on your particular 3.14 setup, use Python 3.12 and replace `py -3.14` with `py -3.12`. Install exactly one OpenCV variant: this app needs `opencv-python` for its window; do not also install `opencv-python-headless`. See [ONNX Runtime wheels](https://pypi.org/project/onnxruntime/), [PyTorch's Windows install selector](https://pytorch.org/get-started/locally/), [Ultralytics install guide](https://docs.ultralytics.com/quickstart/), [EasyOCR install notes](https://pypi.org/project/easyocr/), and [pyttsx3 compatibility classifiers](https://pypi.org/project/pyttsx3/).

Optional scene descriptions use Google's Gemini API via Python's built-in HTTP client, so there is no extra SDK to install. Create a key in [Google AI Studio](https://aistudio.google.com/apikey), then set it in PowerShell for the current window with `$env:GEMINI_API_KEY = "your-key"`. Local camera inference and safety alerts remain offline. Without the key or internet, Gemini features say "Description unavailable" and the local safety system continues. Camera images are sent to Google only when `s`, `f`, `r`, or `--auto-describe` triggers a request. The optional legacy Anthropic provider can be selected with `$env:BLIND_ASSIST_LLM = "anthropic"` and `ANTHROPIC_API_KEY`, but requires installing `anthropic` separately.

**OpenCV note after installing EasyOCR:** EasyOCR pulls in `opencv-python-headless`, replacing the GUI build's shared `cv2` files. Restore the window-capable build after OCR installation:

```powershell
py -3.14 -m pip uninstall -y opencv-python-headless
py -3.14 -m pip install --force-reinstall --no-deps opencv-python
```

## Run modes

```powershell
py blind_assist.py
py blind_assist.py --source 1
py blind_assist.py --source http://PHONE-IP:8080/video
py blind_assist.py --model custom.pt --no-depth
py blind_assist.py --small "pen,pencil,keys" --tiling
py blind_assist.py --describe --find "door"
py blind_assist.py --auto-describe 30 --traffic-light --ocr
py blind_assist.py --record demo.mp4 --log run.csv
py blind_assist.py --audio-log audio.csv --audio-sensitivity 0.7 --mic-distance 0.18
py blind_assist.py --audio-test
py blind_assist.py --phone-sensors
```

`--source` accepts a webcam index, video path, or phone IP Webcam URL. `--no-small` disables the optional YOLO-World object layer; `--no-depth` disables MiDaS depth; `--no-audio` disables the microphone sound classifier. YAMNet ONNX weights and labels download to `audio_models/` on first audio use, then inference is local/offline. If audio model download, ONNX Runtime, sounddevice, or the mic is unavailable, audio awareness reports the problem and vision continues. Depth and small-object weights are also downloaded by Ultralytics/PyTorch on first use. Cloud descriptions are requested only when `s`, `f`, `r`, or `--auto-describe` triggers them, and are skipped gracefully without the API key.

`--audio-test` listens for a left/right clap check and then plays left/right beeps without starting a camera. Panning requires `sounddevice` and stereo headphones/speakers; the `winsound` fallback is mono. `--mic-distance` is the spacing between two microphones. `--mic-array 4` requests a four-channel array and uses channel-energy quadrants as a basic 360° direction cue; channel order depends on the array driver. A two-microphone line can estimate left/right but cannot resolve front versus rear reliably. A mono mic always reports direction unknown. YAMNet uses AudioSet labels for general sound-event awareness, so treat confidence and labels as approximate rather than guaranteed hazard detection. The app downloads the ONNX conversion of Google's YAMNet and its class map on first use from [the model card](https://huggingface.co/audiomagic/yamnet-onnx).

The test prints microphone input in dBFS, which is a digital level, not calibrated sound-pressure level. If it stays around `-80` to `-100 dBFS`, open Windows **Settings → System → Sound → Input**, select the intended microphone, raise its input volume, and use the input test while clapping or speaking near it. Also check **Settings → Privacy & security → Microphone** permissions. The test intentionally reports direction unknown and suppresses sound events when input is near silence. The `onnxruntime_test.exe is not on PATH` pip warning is harmless for this app; it imports ONNX Runtime as a library and does not call that executable.

## Keys

- `q`: quit.
- `s`: speak a local summary and request a two-sentence Gemini analysis of approaching tracked objects, object positions, visible people/activity, small items, and the user's locally estimated standing/walking/running state. The app passes YOLO/ByteTrack relative closing speed and TTC as context; the LLM must not invent them. Set `GEMINI_API_KEY` for the optional request.
- `f`: ask for an object name in the console when no `--find` target was provided; with `--find`, ask the cloud find mode for that target. Local detections also report their direction.
- `r`: run OCR if enabled; if OCR is unavailable, request cloud text reading when cloud mode is enabled.

## Threads and safety

The camera reader keeps the newest frame, while local YOLO tracking and TTC assessment run in the main loop. The display annotates multiple detections with distance and approach/away/crossing estimates. MiDaS, optional YOLO-World small-item inference, local auto-rickshaw inference, microphone capture/YAMNet classification, YOLO pose, and Gemini requests each run in background workers using copied frames/latest-result slots. On `s` or `--auto-describe N`, the LLM receives the user's optically estimated standing/walking/running speed plus a TTC-ranked list of local detections with distance, direction, relative closing speed, and TTC. It can add visual descriptions of people and small objects, but doesn't control safety alerts. The optical-flow motion estimate is a rough visual heuristic; `--phone-sensors` optionally listens for UDP JSON such as `{"speed_mps":1.2}` on port 8765. Phone sensor data overrides the flow estimate. Speech and beeps run asynchronously; danger speech preempts queued speech and invalidates an outstanding LLM response. The system is a prototype, not a substitute for a cane, guide dog, or orientation training.

Vehicle alerts for cars, buses, trucks, bicycles, and motorcycles use the local track/TTC safety logic and do not depend on Gemini. Inference defaults to CPU. If you run with `--device cuda:0`, only the main safety YOLO model uses the GPU; YOLO-World, depth, pose, and OCR stay on CPU. For lower CPU load, use `--no-small --no-depth --no-audio --imgsz 416`; optional workers such as YOLO-World, depth, pose, and audio can each add load. A separately running local Roboflow Inference Server controls its own device selection.

## Microphone placement and hardware

For two-mic direction, mount the capsules about 18 cm apart, in one horizontal line, with the line facing forward and the mic labels/channel order noted for left versus right. Keep the spacing stable and unobstructed; use small foam windscreens outdoors and shield the mic from direct wind and clothing rub. Do not put both capsules inside a pocket. Start with a USB stereo microphone or inexpensive two-channel USB sound card plus two matched electret capsules. For four-channel tests, a ReSpeaker 4-Mic array is an option; India listings include [Tech Depot](https://techdepot.in/products/respeaker-4-mic-linear-array-kit-for-raspberry-pi), [Robocraze](https://robocraze.com/products/respeaker-mic-array-v3-0-with-4-mic-array-and-xvf3000-voice-processor-seeed-studio), and [MG Super Labs](https://www.mgsuperlabs.co.in/estore/ReSpeaker-4Mic-Linear-Array-Kit-for-Raspberry-Pi). Check current stock, USB compatibility, and array channel layout before buying. Cheap lapel mics are usually mono and cannot give direction.

## Auto-rickshaw training

COCO weights do not include auto-rickshaw reliably. Label a Delhi traffic dataset in YOLO format and train locally:

```powershell
yolo detect train model=yolov8n.pt data=autorickshaw.yaml epochs=80 imgsz=640
py blind_assist.py --model runs/detect/train/weights/best.pt
```

Use separate train/validation scenes and include varied angles, lighting, occlusion, and vehicle sizes. Keep a custom dataset's license and consent requirements in mind.

## Five common live-demo failures

1. First-run model downloads are slow: cache YOLO, depth, and YAMNet weights before the demo and test once offline.
2. Windows selects a muted or quiet microphone: run `--audio-test`, verify input level, select the right mic, and check mic permissions.
3. EasyOCR reinstalls headless OpenCV: run the GUI OpenCV repair commands above after installing OCR.
4. Optional models compete for CPU/GPU and lower FPS: pre-cache weights and disable unused layers with `--no-small`, `--no-depth`, or `--no-audio`.
5. Wind or noisy traffic causes false or uncertain audio cues: windscreen the mic, secure it about 18 cm from the second capsule, and treat labels and direction as demo estimates.
