#!/usr/bin/env python3
"""Blind Assist: low-latency object awareness and collision cues.

Install (Windows): py -3.14 -m pip install ultralytics opencv-python numpy sounddevice
Optional OCR:      py -3.14 -m pip install easyocr
Local Roboflow:    no Python SDK required by this script (uses local HTTP + stdlib)
Start the local Roboflow Inference Server (Docker Desktop required):
  py -3.12 -m pip install inference-cli
  inference server start --port 9001
Then run with: py blind_assist.py --auto-model auto-rickshaw-9fpsm/3
The server downloads/caches model weights once; camera frames go only to localhost.
Set ROBOFLOW_API_KEY in the environment if the Roboflow project is private.
Depth uses the MiDaS-small weights via torch.hub (first run needs internet and torch).
For best compatibility/performance, Python 3.11-3.13 is currently the safer fallback
if PyTorch/Ultralytics wheels are unavailable for a particular Python 3.14/Windows build.

Custom auto-rickshaw training example (dataset YAML has train/val and names: [auto-rickshaw]):
  yolo detect train model=yolov8n.pt data=autorickshaw.yaml epochs=80 imgsz=640
Then run: py blind_assist.py --model runs/detect/train/weights/best.pt
Collect varied Delhi street images, label tight boxes consistently, and validate on
separate traffic scenes. COCO weights alone cannot recognize auto-rickshaws reliably.

This is a hackathon prototype, not a certified mobility aid. Never rely on it instead
of a cane, guide dog, or orientation training. Camera depth and TTC are rough estimates.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import socket
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

# Tuning: typical object heights seed monocular distance. Delhi auto-rickshaws are ~1.7 m.
KNOWN_HEIGHT_M = {
    "person": 1.70, "bicycle": 1.10, "motorcycle": 1.20, "car": 1.50,
    "auto-rickshaw": 1.70, "autorickshaw": 1.70, "rickshaw": 1.70,
    "bus": 3.00, "truck": 3.00, "train": 3.50, "dog": 0.55, "cat": 0.30,
    "horse": 1.60, "cow": 1.50, "traffic light": 0.80, "stop sign": 0.75,
    "fire hydrant": 0.70, "bench": 0.85, "chair": 0.90, "potted plant": 0.50,
    "suitcase": 0.60, "backpack": 0.45, "umbrella": 0.90, "bottle": 0.25,
    "door": 2.00,
}
DEFAULT_HEIGHT_M = 1.0
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "train", "auto-rickshaw", "autorickshaw", "rickshaw"}
IMPORTANT = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "dog", "cow", "auto-rickshaw", "autorickshaw", "rickshaw"}
ANNOUNCE = set(KNOWN_HEIGHT_M) | IMPORTANT
CORRIDOR_HALF_WIDTH_M = 0.9       # walking corridor, metres
MIN_CLOSING_SPEED = 0.6            # ignore small/noisy relative motion (m/s)
DANGER_TTC, WARNING_TTC = 2.5, 5.0
HISTORY_SECONDS, EMA_ALPHA = 1.0, 0.45
COOLDOWN = {"danger": 1.2, "warning": 4.0, "info": 12.0}
DEPTH_EVERY_N_FRAMES = 3           # depth inference is asynchronous
DEPTH_FUSE_WEIGHT = 0.45           # relative-depth contribution after calibration
OBSTACLE_NEAR_M = 1.5              # conservative central-strip obstacle threshold
SMALL_DEFAULT = "pen,pencil,keys,phone,glasses,cup,bottle,wallet,remote,scissors"
SMALL_EVERY_N_FRAMES = 3            # isolate small-item inference from safety FPS
SMALL_IMGSZ = 768                    # higher resolution helps small objects
LLM_TIMEOUT_S = 10                   # on-demand requests never hold up the camera loop
AUDIO_RATE = 16000                   # YAMNet waveform sample rate
AUDIO_WINDOW_S = 1.0                  # classify one second at a time
AUDIO_HOP_S = 0.5                    # 50% overlap
AUDIO_MODEL_URL = "https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet.onnx"
AUDIO_LABELS_URL = "https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet_class_map.csv"
SPEED_OF_SOUND = 343.0


class AsyncSmallDetector:
    """YOLO-World open-vocabulary detector; runs on a copied frame in its own worker."""
    def __init__(self, classes, model_path="yolov8s-worldv2.pt", imgsz=SMALL_IMGSZ,
                 tiling=False, device=None):
        self.classes = [x.strip() for x in classes.split(",") if x.strip()]
        self.model_path, self.imgsz, self.tiling, self.device = model_path, imgsz, tiling, device
        self.lock = threading.Lock(); self.pending = None; self.latest = (0.0, [])
        self.enabled = bool(self.classes); self.seen = set(); self.ready = False
        if self.enabled: threading.Thread(target=self._worker, daemon=True).start()
    def submit(self, frame, now):
        if self.enabled:
            with self.lock: self.pending = (frame.copy(), now)
    def get(self):
        with self.lock: return self.latest
    def _worker(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path); model.set_classes(self.classes); self.ready = True
        except Exception as e:
            self.enabled = False; print(f"[small-object detector unavailable: {e}]"); return
        while self.enabled:
            with self.lock: item, self.pending = self.pending, None
            if item is None: time.sleep(.01); continue
            frame, ts = item
            try:
                # Optional lightweight 2x overlapping tiles; globalize boxes after inference.
                tiles = [(frame, 0, 0)]
                if self.tiling:
                    h,w=frame.shape[:2]; th,tw=int(h*.65),int(w*.65)
                    tiles=[(frame[y:min(h,y+th),x:min(w,x+tw)],x,y)
                           for y in (0,max(0,h-th)) for x in (0,max(0,w-tw))]
                found=[]
                for tile,ox,oy in tiles:
                    result=model.predict(tile, imgsz=self.imgsz, conf=.22, verbose=False,
                                         device=self.device)[0]
                    if result.boxes is None: continue
                    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                               result.boxes.cls.int().cpu().tolist(),
                                               result.boxes.conf.cpu().tolist()):
                        found.append((tuple(map(float,(box[0]+ox,box[1]+oy,box[2]+ox,box[3]+oy))),
                                      str(model.names[cls]),float(conf)))
                with self.lock: self.latest=(ts,found)
            except Exception as e: print(f"[small-object inference error: {e}]"); time.sleep(.2)


class AsyncPose:
    """Low-rate optional pose worker; a basic pose cue annotates people, never drives alerts."""
    def __init__(self, device=None):
        self.device=device; self.lock=threading.Lock(); self.pending=None; self.latest=(0.0,[])
        threading.Thread(target=self._worker,daemon=True).start()
    def submit(self,frame,now):
        with self.lock: self.pending=(frame.copy(),now)
    def get(self):
        with self.lock: return self.latest
    def _worker(self):
        try:
            from ultralytics import YOLO
            model=YOLO("yolov8n-pose.pt")
        except Exception as e: print(f"[pose refinement unavailable: {e}]"); return
        while True:
            with self.lock: item,self.pending=self.pending,None
            if item is None: time.sleep(.02); continue
            frame,ts=item
            try:
                res=model.predict(frame,imgsz=416,conf=.4,verbose=False,device=self.device)[0]
                cues=[]
                if res.keypoints is not None:
                    for box,kps in zip(res.boxes.xyxy.cpu().numpy(),res.keypoints.xy.cpu().numpy()):
                        if len(kps)<17: continue
                        # COCO keypoints: shoulders 5/6, wrists 9/10, hips 11/12, knees 13/14.
                        shoulder=np.mean(kps[[5,6]],axis=0); wrists=kps[[9,10]]; hips=np.mean(kps[[11,12]],axis=0); knees=np.mean(kps[[13,14]],axis=0)
                        cue="sitting" if np.linalg.norm(hips-knees)<max(25,frame.shape[0]*.07) else "standing/walking"
                        if np.mean(wrists[:,1])<shoulder[1]: cue+="; arm raised"
                        cues.append((tuple(map(float,box)),cue))
                with self.lock: self.latest=(ts,cues)
            except Exception as e: print(f"[pose inference error: {e}]")


class AsyncDescribe:
    """Optional Gemini (default) or Anthropic calls; never blocks local vision."""
    def __init__(self):
        # Gemini uses the standard-library HTTP client, so no cloud SDK is required.
        self.provider=os.environ.get("BLIND_ASSIST_LLM", "gemini").lower()
        self.key=os.environ.get("GEMINI_API_KEY") if self.provider == "gemini" else os.environ.get("ANTHROPIC_API_KEY")
        self.lock=threading.Lock()
        self.pending=None; self.result=None; self.busy=False; self.generation=0
        threading.Thread(target=self._worker,daemon=True).start()
    def submit(self, frame, context, task="describe", target=""):
        if not self.key:
            with self.lock: self.result=(task,"Description unavailable.")
            return
        with self.lock:
            if not self.busy:
                self.generation+=1; self.pending=(frame.copy(),context,task,target,self.generation); self.busy=True
    def describe(self, frame, context):
        """Queue a default two-sentence scene description without blocking the caller."""
        self.submit(frame,context,"describe")
    def cancel(self):
        with self.lock:
            self.generation+=1; self.pending=None; self.result=None
    def pop(self):
        with self.lock: r,self.result=self.result,None; return r
    def _worker(self):
        while True:
            with self.lock: item,self.pending=self.pending,None
            if item is None: time.sleep(.03); continue
            frame,context,task,target,generation=item
            try:
                h,w=frame.shape[:2]; nw=640; small=cv2.resize(frame,(nw,max(1,int(h*nw/w))))
                ok,buf=cv2.imencode(".jpg",small,[cv2.IMWRITE_JPEG_QUALITY,72])
                if not ok: raise RuntimeError("JPEG encoding failed")
                prompt={"describe":"Describe the live scene in exactly two short sentences. First state the user's measured motion state, then prioritize tracked objects genuinely approaching, with their left/center/right position and TTC when supplied. Next mention what visible people appear to be doing and any visible small items. Local context contains measured relative motion: say 'approaching' only for positive closing speed and do not invent distances, speeds, directions, identities, or user actions. If no tracked object is approaching, say so.",
                        "find":f"Where is {target} relative to the user? Use left, right, or ahead, and say not visible if absent. Do not invent distances; use local context.",
                        "read":"Read visible text/signs in this image. Be concise."}[task]
                image_b64=base64.b64encode(buf).decode("ascii")
                full_prompt=prompt+"\nLocal context (prefer these measurements; do not guess): "+context
                if self.provider == "gemini":
                    url=("https://generativelanguage.googleapis.com/v1beta/models/"
                         "gemini-3.8-flash:generateContent?key="+urllib.parse.quote(self.key, safe=""))
                    payload={"contents":[{"parts":[{"text":full_prompt},
                        {"inline_data":{"mime_type":"image/jpeg","data":image_b64}}]}],
                        "generationConfig":{"maxOutputTokens":180,"temperature":0.2}}
                    request=urllib.request.Request(url,data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type":"application/json"},method="POST")
                    with urllib.request.urlopen(request,timeout=LLM_TIMEOUT_S) as response:
                        data=json.loads(response.read().decode("utf-8"))
                    answer=" ".join(p.get("text","") for p in data.get("candidates",[{}])[0]
                        .get("content",{}).get("parts",[])).strip() or "Description unavailable."
                elif self.provider == "anthropic":
                    import anthropic
                    client=anthropic.Anthropic(api_key=self.key,timeout=LLM_TIMEOUT_S)
                    response=client.messages.create(model="claude-haiku-5-5",max_tokens=180,
                        messages=[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":image_b64}},
                            {"type":"text","text":full_prompt}]}])
                    answer=" ".join(getattr(b,"text","") for b in response.content).strip() or "Description unavailable."
                else:
                    raise RuntimeError("BLIND_ASSIST_LLM must be gemini or anthropic")
                if task=="find" and target:
                    locally_seen=bool(re.search(r"\b"+re.escape(target.lower())+r"\b",context.lower()))
                    answer += (f" Local detections include {target}." if locally_seen else
                               f" Local detection has not confirmed {target}; treat this location as unverified.")
            except Exception as e:
                answer="Description unavailable."; print(f"[cloud description unavailable: {e}]")
            with self.lock:
                if generation==self.generation: self.result=(task,answer)
                self.busy=False


class PhoneSensorReceiver:
    """Optional UDP JSON receiver: send {\"speed_mps\": 1.2} to the listen port."""
    def __init__(self, port=8765):
        self.speed=None
        try:
            self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self.sock.bind(("0.0.0.0",port)); self.sock.settimeout(.5)
            threading.Thread(target=self._loop,daemon=True).start(); print(f"[phone sensor UDP listening on {port}]")
        except Exception as e: self.sock=None; print(f"[phone sensor receiver unavailable: {e}]")
    def _loop(self):
        while True:
            try:
                data,_=self.sock.recvfrom(2048); value=json.loads(data.decode()).get("speed_mps")
                if value is not None: self.speed=float(np.clip(float(value),0,8))
            except socket.timeout: pass
            except Exception: pass


class Speaker:
    """Speech worker: pyttsx3 first, native Windows SAPI fallback, never fatal."""
    PRIORITY = {"danger": 0, "warning": 1, "info": 2}

    def __init__(self):
        self.q = queue.PriorityQueue()
        self.counter = 0
        self.engine = None
        self.current_priority = 99
        self.process = None
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 190)
        except Exception as e:
            print(f"[speech: using Windows SAPI fallback if available: {e}]")
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text, level="info"):
        print(f"[{level.upper()}] {text}")
        if level == "danger":
            self._drain()
            if self.engine is not None and self.current_priority > 0:
                try: self.engine.stop()
                except Exception: pass
            if self.process is not None and self.current_priority > 0:
                try: self.process.terminate()
                except Exception: pass
        elif level == "warning" and self.q.qsize() > 2: return
        elif level == "info" and self.q.qsize() > 0: return
        self.counter += 1
        self.q.put((self.PRIORITY[level], self.counter, text))

    def _drain(self):
        try:
            while True: self.q.get_nowait()
        except queue.Empty: pass

    def _worker(self):
        while True:
            prio, _, text = self.q.get()
            self.current_priority = prio
            try:
                if self.engine is not None:
                    self.engine.say(text); self.engine.runAndWait()
                elif os.name == "nt":
                    escaped = text.replace("'", "''")
                    script = ("Add-Type -AssemblyName System.Speech; "
                              "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                              f"$s.Speak('{escaped}')")
                    self.process=subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    self.process.wait(timeout=20); self.process=None
            except Exception as e:
                print(f"[speech unavailable: {e}]")
            self.current_priority = 99


class Beeper:
    """Non-blocking stereo risk beeps; sounddevice preferred, Windows winsound fallback."""
    def __init__(self):
        self.sd = None
        try:
            import sounddevice as sd
            self.sd = sd
        except Exception: pass
        self.lock = threading.Lock()
        self.last = 0.0
    def cue(self, x_norm, ttc, danger=False):
        now=time.monotonic(); interval=.18 if danger else (.32 if ttc is not None and ttc<3 else .65)
        if now-self.last<interval: return
        self.last=now
        threading.Thread(target=self._play,args=(x_norm,danger),daemon=True).start()
    def _play(self,pan,danger):
        try:
            if self.sd is not None:
                rate,dur=22050,(.10 if danger else .07); t=np.arange(int(rate*dur))/rate
                wave=(.20*np.sin(2*np.pi*(1000 if danger else 760)*t)*np.hanning(len(t))).astype(np.float32)
                left=math.sqrt(max(0,(1-pan)/2)); right=math.sqrt(max(0,(1+pan)/2))
                self.sd.play(np.column_stack((wave*left,wave*right)),rate,blocking=False)
            elif os.name=="nt":
                import winsound
                winsound.Beep(1000 if danger else 760,90 if danger else 60)
        except Exception: pass


class AudioMonitor:
    """Best-effort local YAMNet microphone monitor. Capture/inference never run in video loop."""
    IMPORTANT_MAP = {
        "car horn": ("Car horn", ("car horn", "vehicle horn", "honking")),
        "siren": ("Siren", ("siren",)),
        "engine": ("Vehicle engine", ("engine", "motor vehicle", "vehicle")),
        "motorcycle": ("Motorcycle", ("motorcycle", "motorbike")),
        "truck": ("Truck", ("truck", "heavy engine")),
        "bus": ("Bus", ("bus",)),
        "bicycle bell": ("Bicycle bell", ("bicycle bell", "bicycle")),
        "train": ("Train", ("train", "rail transport")),
        "tire squeal": ("Tire squeal", ("tire squeal", "skidding", "skid")),
        "dog bark": ("Dog bark", ("dog", "bark", "bow-wow")),
        "shouting": ("Shouting", ("shout", "yell", "scream")),
        "footsteps": ("Footsteps", ("footsteps", "walk, footsteps")),
        "alarm": ("Alarm", ("alarm", "smoke detector", "vehicle alarm")),
    }
    def __init__(self, enabled=True, sensitivity=.55, mic_distance=.18, mic_array=0, log_path=None):
        self.enabled=enabled; self.sensitivity=float(np.clip(sensitivity,0,1))
        self.mic_distance=max(.04,float(mic_distance)); self.mic_array=mic_array
        self.latest={"label":"","confidence":0.0,"direction":"direction unknown","trend":"constant","db":0.0,"distance":"far","time":0.0,"error":""}
        self.lock=threading.Lock(); self.q=queue.Queue(maxsize=12); self.running=enabled
        self.log_path=log_path; self.db_history=deque(maxlen=4); self.last_label=""; self.last_event=0
        self.channels=0; self._error_reported=False
        self.ready=False
        self.noise_floor_db=None
        if enabled: threading.Thread(target=self._run,daemon=True).start()
    def get(self):
        with self.lock: return dict(self.latest)
    def _get_assets(self):
        folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),"audio_models")
        os.makedirs(folder,exist_ok=True); model_path=os.path.join(folder,"yamnet.onnx"); labels_path=os.path.join(folder,"yamnet_class_map.csv")
        for path,url in ((model_path,AUDIO_MODEL_URL),(labels_path,AUDIO_LABELS_URL)):
            if not os.path.exists(path):
                print(f"[audio] downloading local YAMNet asset: {os.path.basename(path)}")
                temp_path=path+".part"
                with urllib.request.urlopen(url,timeout=25) as response, open(temp_path,"wb") as target:
                    while True:
                        block=response.read(1024*1024)
                        if not block: break
                        target.write(block)
                os.replace(temp_path,path)
        return model_path,labels_path
    @staticmethod
    def _highpass(x,rate):
        # One-pole high-pass, approximately 80 Hz: rejects wind/handling rumble.
        alpha=math.exp(-2*math.pi*80/rate); y=np.empty_like(x); previous_x=previous_y=0.0
        for i,value in enumerate(x):
            previous_y=alpha*(previous_y+float(value)-previous_x); previous_x=float(value); y[i]=previous_y
        rms=float(np.sqrt(np.mean(y*y)+1e-12)); gain=float(np.clip(.08/max(rms,1e-4),1,10))
        return np.clip(y*gain,-1,1).astype(np.float32),rms
    @staticmethod
    def _gcc_delay(a,b,rate):
        n=min(len(a),len(b)); a=a[:n]-np.mean(a[:n]); b=b[:n]-np.mean(b[:n])
        nfft=1 << int(math.ceil(math.log2(max(2,2*n-1))))
        cross=np.fft.rfft(a,nfft)*np.conj(np.fft.rfft(b,nfft)); cross/=np.maximum(np.abs(cross),1e-12)
        corr=np.fft.irfft(cross,nfft); corr=np.concatenate((corr[-nfft//2:],corr[:nfft//2]))
        maxlag=min(int(rate*.0006),len(corr)//2-1); center=len(corr)//2
        lag=int(np.argmax(corr[center-maxlag:center+maxlag+1])-maxlag)
        return lag/rate
    def _direction(self,channels,rate):
        if channels.ndim<2 or channels.shape[1]<2: return "direction unknown",0.0
        rms0=float(np.sqrt(np.mean(channels[:,0]**2)+1e-12)); rms1=float(np.sqrt(np.mean(channels[:,1]**2)+1e-12))
        corr=float(np.corrcoef(channels[:,0],channels[:,1])[0,1]) if rms0>1e-6 and rms1>1e-6 else 0.0
        if abs(corr)>.9998 and abs(rms0-rms1)<1e-5: return "direction unknown",0.0
        if self.mic_array>=4 and channels.shape[1]>=4:
            # ReSpeaker-style four-channel input: compare channel energy by configured array order.
            energy=np.sqrt(np.mean(channels[:,:4]**2,axis=0)); idx=int(np.argmax(energy))
            names=("ahead","right","behind you","left")
            return names[idx],float(energy[idx]/max(np.sum(energy),1e-9))
        tau=self._gcc_delay(channels[:,0],channels[:,1],rate)
        # A two-element line array only resolves left/right; front/back is physically ambiguous.
        sine=float(np.clip(tau*SPEED_OF_SOUND/self.mic_distance,-1,1)); angle=math.degrees(math.asin(sine))
        sector="on your left" if angle < -50 else "front-left" if angle < -12 else "ahead" if angle <= 12 else "front-right" if angle <= 50 else "on your right"
        return sector,abs(sine)
    def _run(self):
        try:
            import sounddevice as sd
            import onnxruntime as ort
            model_path,labels_path=self._get_assets()
            session=ort.InferenceSession(model_path,providers=["CPUExecutionProvider"])
            input_name=session.get_inputs()[0].name
            with open(labels_path,newline="",encoding="utf-8") as f:
                reader=csv.DictReader(f); labels=[row.get("display_name",row.get("name","")) for row in reader]
            device=sd.query_devices(kind="input"); maxch=int(device.get("max_input_channels",1))
            requested=4 if self.mic_array>=4 else 2
            channels=min(requested,maxch) if maxch>0 else 1
            stream_rate=int(device.get("default_samplerate",48000))
            def callback(indata,frames,time_info,status):
                if status: pass
                try: self.q.put_nowait((time.monotonic(),indata.copy()))
                except queue.Full:
                    try: self.q.get_nowait(); self.q.put_nowait((time.monotonic(),indata.copy()))
                    except queue.Empty: pass
            try:
                stream=sd.InputStream(samplerate=stream_rate,channels=channels,dtype="float32",blocksize=max(1,int(stream_rate*AUDIO_HOP_S)),callback=callback)
                stream.start()
            except Exception as first_error:
                if channels==1: raise
                channels=1
                stream=sd.InputStream(samplerate=stream_rate,channels=1,dtype="float32",blocksize=max(1,int(stream_rate*AUDIO_HOP_S)),callback=callback)
                stream.start(); print(f"[audio] stereo unavailable; using mono ({first_error})")
            self.channels=channels
            self.ready=True
            print(f"[audio] YAMNet ready; mic={device.get('name','default')}, channels={channels}, rate={stream_rate} Hz")
            pending=np.zeros((0,channels),dtype=np.float32); pending_start=None
            hop=int(stream_rate*AUDIO_HOP_S); window=int(stream_rate*AUDIO_WINDOW_S)
            while self.running:
                try:
                    chunk_time,chunk=self.q.get(timeout=.2)
                    if pending_start is None: pending_start=chunk_time
                    pending=np.concatenate((pending,chunk),axis=0)
                except queue.Empty: continue
                if len(pending)<window: continue
                raw=pending[:window].copy(); pending=pending[hop:]
                capture_end=(pending_start or time.monotonic())+AUDIO_WINDOW_S
                pending_start=(pending_start or time.monotonic())+AUDIO_HOP_S
                mono=np.mean(raw,axis=1).astype(np.float32)
                # Resample to YAMNet's 16 kHz with dependency-free linear interpolation.
                src=np.arange(len(mono),dtype=np.float32); dst=np.linspace(0,len(mono)-1,int(AUDIO_RATE*AUDIO_WINDOW_S),dtype=np.float32)
                mono16=np.interp(dst,src,mono).astype(np.float32)
                filtered,rms=self._highpass(mono16,AUDIO_RATE)
                input_shape=session.get_inputs()[0].shape
                feed=filtered.reshape(1,-1) if len(input_shape)==2 else filtered
                out=session.run(None,{input_name:feed})[0]
                scores=np.asarray(out).reshape(-1,len(labels)).mean(axis=0)
                eligible=[]
                for class_idx,name in enumerate(labels):
                    lower=name.lower()
                    for key,(display,terms) in self.IMPORTANT_MAP.items():
                        if any(term in lower for term in terms): eligible.append((float(scores[class_idx]),display)); break
                confidence,label=max(eligible,default=(0.0,""))
                db=20*math.log10(max(rms,1e-6)); self.db_history.append((time.monotonic(),db))
                # Slow rolling floor follows ambient changes, but does not rise on loud transients.
                if self.noise_floor_db is None: self.noise_floor_db=db
                elif db < self.noise_floor_db+10: self.noise_floor_db=.96*self.noise_floor_db+.04*db
                relative_db=db-self.noise_floor_db
                threshold=float(np.clip(.62-.42*self.sensitivity-.025*max(0,relative_db-8),.14,.62))
                trend="constant"
                if len(self.db_history)>=3 and self.db_history[-1][0]-self.db_history[0][0]>=1.0:
                    slope=(self.db_history[-1][1]-self.db_history[0][1])/max(self.db_history[-1][0]-self.db_history[0][0],.1)
                    trend="getting louder" if slope>2.5 else "fading" if slope< -2.5 else "constant"
                direction,angle_quality=self._direction(raw,stream_rate)
                # Suppress bearings/classifications from electrical hiss and near-silence.
                # -55 dBFS is an approximate input-quality threshold, not calibrated SPL.
                if db < -55.0: direction,angle_quality="direction unknown",0.0
                if db < -65.0: confidence,label=0.0,""
                distance="near" if relative_db>28 else "medium" if relative_db>14 else "far"
                if confidence<threshold: label=""
                now=time.time()
                with self.lock:
                    self.latest={"label":label,"confidence":confidence,"direction":direction,"trend":trend,"db":db,"noise_floor_db":self.noise_floor_db,"mic_level":rms,"distance":distance,"time":now,"captured_monotonic":capture_end,"error":"","angle_quality":angle_quality}
                if label and (label!=self.last_label or now-self.last_event>3.0):
                    self.last_label=label; self.last_event=now
        except Exception as e:
            with self.lock: self.latest["error"]=str(e)
            if not self._error_reported:
                print(f"[audio awareness unavailable: {e}]")
                print("Install sounddevice and onnxruntime, connect a microphone, and allow first-run YAMNet download; vision safety continues.")
                self._error_reported=True
    def stop(self): self.running=False


class LatestFrameReader:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open video source: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.ok, self.lock = None, True, threading.Lock()
        self.is_file = isinstance(source, str) and not source.lower().startswith(("http://", "https://", "rtsp://"))
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self.ok:
            ok, frame = self.cap.read()
            if not ok: self.ok = False; break
            with self.lock: self.frame = frame
            if self.is_file: time.sleep(1 / 30)
    def read(self):
        with self.lock: f, self.frame = self.frame, None
        return f
    def release(self): self.ok = False; self.cap.release()


class AsyncDepth:
    """MiDaS-small worker. Its output is relative depth; calibration aligns to box ranges."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.pending = None
        self.depth = None
        self.error = None
        self.ready = threading.Event()
        if enabled: threading.Thread(target=self._worker, daemon=True).start()
    def submit(self, frame):
        if not self.enabled: return
        with self.lock: self.pending = frame.copy()
    def _worker(self):
        model = transform = None
        while True:
            with self.lock: frame, self.pending = self.pending, None
            if frame is None: time.sleep(0.01); continue
            try:
                if model is None:
                    import torch
                    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                    model.eval()
                    transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
                import torch
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inp = transform(rgb)
                with torch.no_grad(): pred = model(inp)
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
                # Convert inverse/relative depth into a normalized near=1, far=0 map.
                lo, hi = np.percentile(pred, [5, 95])
                norm = np.clip((pred - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
                with self.lock: self.depth = norm; self.error = None
                self.ready.set()
            except Exception as e:
                with self.lock: self.error = str(e)
                self.enabled = False
                print(f"[depth disabled: {e}]")
    def get(self):
        with self.lock: return None if self.depth is None else self.depth.copy()


class AsyncRoboflowDetector:
    """Optional second detector backed by a LOCAL Roboflow Inference Server.

    The server fetches/caches model weights during setup. Camera frames are POSTed only
    to localhost; they are not sent to Roboflow's hosted inference endpoint.
    """
    def __init__(self, model_id, server_url="http://127.0.0.1:9001"):
        self.model_id = model_id
        self.server_url = server_url.rstrip("/")
        self.api_key = os.environ.get("ROBOFLOW_API_KEY")
        self.lock = threading.Lock()
        self.pending = None
        self.latest = (0.0, [])
        self.enabled = bool(model_id)
        self.warned_error = False
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, frame, timestamp):
        if not self.enabled: return
        with self.lock: self.pending = (frame.copy(), timestamp)

    def get(self):
        with self.lock: return self.latest

    def _worker(self):
        while self.enabled:
            with self.lock: item, self.pending = self.pending, None
            if item is None:
                time.sleep(.01)
                continue
            frame, timestamp = item
            try:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok: raise RuntimeError("Could not encode camera frame")
                model_path = urllib.parse.quote(self.model_id.strip("/"), safe="/")
                query = {"confidence": "40"}
                if self.api_key: query["api_key"] = self.api_key
                request = urllib.request.Request(
                    f"{self.server_url}/{model_path}?{urllib.parse.urlencode(query)}",
                    data=base64.b64encode(encoded.tobytes()),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=12) as response:
                    result = json.loads(response.read().decode("utf-8"))
                predictions = []
                for p in result.get("predictions", []):
                    cx, cy = float(p["x"]), float(p["y"])
                    bw, bh = float(p["width"]), float(p["height"])
                    label = str(p.get("class", "auto-rickshaw")).strip().lower().replace(" ", "-")
                    box = (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
                    predictions.append((box, label, float(p.get("confidence", 0.0))))
                with self.lock: self.latest = (timestamp, predictions)
            except Exception as e:
                if not self.warned_error:
                    print(f"[local Roboflow inference unavailable: {e}]")
                    print("Start the local Inference Server and check the model ID; set ROBOFLOW_API_KEY only if the model is private.")
                    self.warned_error = True
                # Retry with the newest frame; startup order no longer permanently disables it.
                time.sleep(2.0)


@dataclass
class Track:
    tid: int
    label: str
    hist: deque = field(default_factory=lambda: deque(maxlen=40))
    z_smooth: float | None = None
    last_seen: float = 0.0
    last_alert: dict = field(default_factory=dict)
    last_level: str = "none"
    announced: bool = False
    closing: float = 0.0
    lateral_v: float = 0.0
    x: float = 0.0
    ttc: float | None = None
    path_x_at_hit: float | None = None
    bbox: tuple = (0, 0, 0, 0)
    alert_started: float | None = None


def fit_slope(ts, vals):
    if len(ts) < 4 or ts[-1] - ts[0] < 0.25: return 0.0
    t0 = ts - ts.mean(); denom = float((t0 ** 2).sum())
    return float((t0 * (vals - vals.mean())).sum() / denom) if denom > 1e-9 else 0.0


def update_motion(tr, now, own_forward_speed=0.0):
    pts = [p for p in tr.hist if now - p[0] <= HISTORY_SECONDS]
    if len(pts) < 4: tr.closing = tr.lateral_v = 0.0; tr.ttc = None; return
    arr = np.array(pts)
    raw_closing = -fit_slope(arr[:, 0], arr[:, 1])
    # Camera moving forward makes static scenery appear to approach: subtract estimated ego speed.
    tr.closing = raw_closing - own_forward_speed
    tr.lateral_v = fit_slope(arr[:, 0], arr[:, 2])
    z = tr.z_smooth or arr[-1, 1]; tr.x = float(arr[-1, 2])
    if tr.closing > MIN_CLOSING_SPEED:
        tr.ttc = z / tr.closing; tr.path_x_at_hit = tr.x + tr.lateral_v * tr.ttc
    else: tr.ttc = tr.path_x_at_hit = None


def activity(tr):
    speed = math.hypot(tr.closing, tr.lateral_v)
    if tr.label == "person": return "standing" if speed < .4 else "walking" if speed < 2.2 else "running"
    if tr.label in VEHICLES: return "stopped" if speed < .8 else "moving slowly" if speed < 6 else "moving fast"
    return "still" if speed < .5 else "moving"


def side_word(x, z):
    ang = math.degrees(math.atan2(x, max(z, .1)))
    return "on your left" if ang < -25 else "front left" if ang < -8 else "ahead" if ang <= 8 else "front right" if ang <= 25 else "on your right"


def assess(tr):
    z = tr.z_smooth or 99.0
    in_lane = abs(tr.x) < CORRIDOR_HALF_WIDTH_M
    will_hit = tr.ttc is not None and tr.path_x_at_hit is not None and abs(tr.path_x_at_hit) < CORRIDOR_HALF_WIDTH_M + (.3 if tr.label in VEHICLES else 0)
    if will_hit and tr.ttc < DANGER_TTC: return "danger", True
    if tr.label in VEHICLES and tr.closing > 2 and tr.ttc is not None and tr.ttc < WARNING_TTC and will_hit: return "danger", True
    if will_hit and tr.ttc < WARNING_TTC: return "warning", True
    if in_lane and z < 1.3 and tr.label not in ("traffic light", "stop sign"): return "warning", False
    if tr.label in VEHICLES and tr.closing > MIN_CLOSING_SPEED and z < 12: return "warning", will_hit
    if z < 4 and tr.label in IMPORTANT: return "info", False
    return "none", will_hit


def build_message(tr, level, will_hit):
    z = tr.z_smooth or 0
    parts = ["Danger! Stop!" if level == "danger" else ""]
    parts.append(f"{tr.label} {side_word(tr.x, z)}")
    parts.append(f"about {z:.0f} meters" if z >= 2 else "very close")
    if tr.closing > MIN_CLOSING_SPEED: parts.append("approaching")
    if tr.label == "person":
        parts.append(f"{activity(tr)}, {getattr(tr,'direction','moving')}")
    if level != "info" and tr.ttc is not None: parts.append(f"in {tr.ttc:.0f} seconds")
    return ", ".join(x for x in parts if x)


def traffic_color(frame, box):
    x1, y1, x2, y2 = map(int, box); h, w = frame.shape[:2]
    roi = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if roi.size == 0: return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # bright saturated pixels approximate illuminated signal lamps
    red = cv2.inRange(hsv, (0, 90, 100), (12, 255, 255)) | cv2.inRange(hsv, (168, 90, 100), (180, 255, 255))
    green = cv2.inRange(hsv, (35, 70, 80), (95, 255, 255))
    nr, ng = cv2.countNonZero(red), cv2.countNonZero(green)
    if max(nr, ng) < max(2, roi.shape[0] * roi.shape[1] * .01): return None
    return "red" if nr > ng else "green"


def main():
    ap = argparse.ArgumentParser(description="Real-time assistive scene awareness")
    ap.add_argument("--source", default="0", help="camera index, video path, or IP Webcam URL (e.g. http://PHONE:8080/video)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights; custom.pt is supported")
    ap.add_argument("--auto-model", default="", help="optional Roboflow model ID, e.g. auto-rickshaw-9fpsm/3; uses the local Inference Server")
    ap.add_argument("--local-inference-url", default="http://127.0.0.1:9001", help="local Roboflow Inference Server URL (never use the hosted URL for camera privacy)")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=.40)
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--device", default="cpu", help="main safety YOLO device: cpu (default) or cuda:0; optional models stay on CPU")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--summary-every", type=float, default=0.0)
    ap.add_argument("--no-depth", action="store_true", help="disable MiDaS-small depth worker")
    ap.add_argument("--traffic-light", action="store_true", help="HSV red/green estimate in detected traffic light boxes")
    ap.add_argument("--ocr", action="store_true", help="enable optional EasyOCR; press r to read visible text")
    ap.add_argument("--find", metavar="OBJECT", help="announce the direction and distance of this object")
    ap.add_argument("--log", nargs="?", const="blind_assist_log.csv", help="write per-frame object metrics CSV")
    ap.add_argument("--record", nargs="?", const="blind_assist_demo.mp4", help="save annotated demo video")
    ap.add_argument("--evaluate", action="store_true", help="interactive 2/4/6 m distance calibration and summary")
    ap.add_argument("--small", default=SMALL_DEFAULT, help="comma-separated YOLO-World vocabulary")
    ap.add_argument("--small-model", default="yolov8s-worldv2.pt")
    ap.add_argument("--small-imgsz", type=int, default=SMALL_IMGSZ)
    ap.add_argument("--tiling", action="store_true", help="overlapping tiles for small-object detector")
    ap.add_argument("--no-small", action="store_true", help="disable YOLO-World small-object layer")
    ap.add_argument("--describe", action="store_true", help="enable optional Gemini scene descriptions (GEMINI_API_KEY)")
    ap.add_argument("--auto-describe", type=float, default=0, metavar="SECONDS")
    ap.add_argument("--phone-sensors", action="store_true", help="listen for UDP JSON speed_mps on port 8765")
    ap.add_argument("--pose", action="store_true", help="use YOLO pose model for optional person pose cues")
    ap.add_argument("--no-audio", action="store_true", help="disable local microphone/YAMNet sound awareness")
    ap.add_argument("--audio-sensitivity", type=float, default=.55, help="sound event sensitivity from 0.0 to 1.0")
    ap.add_argument("--mic-distance", type=float, default=.18, help="spacing in metres between the two stereo mics")
    ap.add_argument("--mic-array", type=int, nargs="?", const=4, default=0, help="use a 4-channel mic array (optionally specify 4)")
    ap.add_argument("--audio-log", nargs="?", const="blind_assist_audio.csv", help="write timestamped sound events to CSV")
    ap.add_argument("--audio-test", action="store_true", help="check mic direction and left/right output; no camera needed")
    args = ap.parse_args()

    speaker, beeper = Speaker(), Beeper()
    if args.audio_test:
        monitor=AudioMonitor(True,args.audio_sensitivity,args.mic_distance,args.mic_array)
        print("Waiting up to 20 s for the mic and local audio model to load.")
        ready_deadline=time.time()+20
        while not monitor.ready and not monitor.get().get("error") and time.time()<ready_deadline: time.sleep(.1)
        if not monitor.ready: print("Mic/model did not become ready: "+(monitor.get().get("error") or "timed out"))
        print("Mic direction check: clap or speak first from the LEFT, then from the RIGHT. Listening for 6 seconds.")
        seen_audio_time=0.0; deadline=time.time()+6.0
        while time.time()<deadline:
            sample=monitor.get()
            if sample.get("time",0)>seen_audio_time:
                seen_audio_time=sample["time"]
                print(f"mic direction: {sample.get('direction','direction unknown')} | level {sample.get('db',0):.0f} dB | event {sample.get('label') or 'none'}")
            time.sleep(.1)
        monitor.stop()
        print("Output pan check: left beep, then right beep. Use stereo headphones/speakers to verify.")
        beeper.cue(-1.0, None); time.sleep(.7); beeper.cue(1.0, None); time.sleep(.7)
        if beeper.sd is None: print("winsound fallback is mono; panning requires sounddevice and stereo output.")
        return
    device = args.device or "cpu"
    if str(device).lower().startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("[GPU unavailable to this PyTorch install; using CPU. Install CUDA-enabled PyTorch only if this laptop has an NVIDIA GPU.]")
                device = "cpu"
        except Exception as e:
            print(f"[GPU check failed ({e}); using CPU]")
            device = "cpu"
    from ultralytics import YOLO
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model); names = model.names
    try: reader = LatestFrameReader(source)
    except Exception as e:
        print(f"[camera unavailable: {e}; no video safety feed can start]"); return
    depth_worker = AsyncDepth(not args.no_depth)
    auto_detector = AsyncRoboflowDetector(args.auto_model, args.local_inference_url)
    # Only the main safety detector may use CUDA. Keep auxiliary models on CPU to limit GPU load.
    small_detector = AsyncSmallDetector("" if args.no_small else args.small, args.small_model,
                                        args.small_imgsz, args.tiling, "cpu")
    pose_detector = AsyncPose("cpu") if args.pose else None
    # Local safety stays offline; Gemini is called only on request and needs GEMINI_API_KEY.
    describer = AsyncDescribe()
    sensors = PhoneSensorReceiver() if args.phone_sensors else None
    audio_monitor=AudioMonitor(not args.no_audio,args.audio_sensitivity,args.mic_distance,args.mic_array,args.audio_log)
    display_enabled = not args.no_display
    tracks = {}
    prev_gray = None; prev_detection_boxes=[]; fps_s = 0.; last_t = time.time(); last_summary = last_t
    own_speed = 0.0; motion_state="standing"; frame_no = 0; obstacle_last = 0.; find_last = 0.
    last_auto_description=0.; latest_context="No objects detected."; find_target=args.find or ""
    csv_file = open(args.log, "w", newline="", encoding="utf-8-sig") if args.log else None
    writer = csv.writer(csv_file) if csv_file else None
    if writer: writer.writerow(["timestamp", "frame", "object_id", "class", "distance_m", "ttc_s", "alert_level", "alert_latency_s"])
    audio_csv=open(args.audio_log,"w",newline="",encoding="utf-8-sig") if args.audio_log else None
    audio_writer=csv.writer(audio_csv) if audio_csv else None
    if audio_writer: audio_writer.writerow(["timestamp","object_id","class","sound","confidence","direction","angle_deg","trend","sound_distance","object_distance_m","ttc_s","dB","visual_match","alert_level","alert_latency_s"])
    last_audio_log_time=0.; last_audio_spoken_time=0.; last_audio_label=""; sound_summary="No sound event"
    video = None; ocr_reader = None; eval_errors = []; eval_lats = []
    if args.ocr:
        try:
            import easyocr
            ocr_reader = easyocr.Reader(["en"], gpu=False)
        except Exception as e: print(f"[OCR unavailable: {e}; install easyocr or use Python 3.11-3.13]")
    speaker.say("Assist started.", "info")

    def summary():
        live = [t for t in tracks.values() if time.time()-t.last_seen < .8 and t.z_smooth]
        if not live: speaker.say("Path looks clear.", "info"); return
        live.sort(key=lambda t:t.z_smooth)
        speaker.say("Around you: " + "; ".join(f"{t.label} {side_word(t.x,t.z_smooth)} {t.z_smooth:.0f} meters" for t in live[:4]), "info")

    eval_targets=[2.0,4.0,6.0]; eval_index=[0]; eval_active=[False]
    if args.evaluate:
        print("Distance evaluation is interactive. At each prompt, place the chosen object at the measured distance and enter its ID shown on screen (or Enter for nearest tracked object).")
        print("A background console prompt will appear for 2, 4, and 6 metres; video and alerts keep running.")
    try:
        while True:
            frame = reader.read()
            if frame is None:
                if not reader.ok: break
                time.sleep(.002); continue
            now = time.time(); frame_no += 1
            h, w = frame.shape[:2]; f_px = (w/2)/math.tan(math.radians(args.hfov/2))
            depth = depth_worker.get()
            if not args.no_depth and frame_no % DEPTH_EVERY_N_FRAMES == 0: depth_worker.submit(frame)
            # Sparse background optical flow: radial expansion gives a conservative ego-motion cue.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=120, qualityLevel=.02, minDistance=12, blockSize=7)
                if pts is not None:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                    good0, good1 = pts[st.ravel()==1].reshape(-1,2), nxt[st.ravel()==1].reshape(-1,2)
                    if len(good0):
                        keep=np.ones(len(good0),dtype=bool)
                        for bx1,by1,bx2,by2 in prev_detection_boxes:
                            keep &= ~((good0[:,0]>=bx1)&(good0[:,0]<=bx2)&(good0[:,1]>=by1)&(good0[:,1]<=by2))
                        good0,good1=good0[keep],good1[keep]
                    if len(good0) >= 12:
                        center = np.array([w/2, h*.62]); r0=np.linalg.norm(good0-center,axis=1)+1; r1=np.linalg.norm(good1-center,axis=1)
                        radial=np.median((r1-r0)/r0)
                        # Keep the estimate bounded; this is only a compensation heuristic.
                        dt=max(now-last_t,1e-3)
                        # Radial flow is only a rough forward-speed cue; phone data overrides it.
                        own_speed=float(np.clip(radial * 1.4 / dt, 0.0, 3.5))
            if sensors is not None and sensors.speed is not None: own_speed=sensors.speed
            motion_state="standing" if own_speed<.25 else "walking" if own_speed<1.8 else "running"
            prev_gray = gray
            auto_detector.submit(frame, now)
            if frame_no % SMALL_EVERY_N_FRAMES == 0: small_detector.submit(frame,now)
            if pose_detector is not None and frame_no % 5 == 0: pose_detector.submit(frame,now)
            res = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=args.imgsz,
                              conf=args.conf, verbose=False, device=device)[0]
            detected_boxes=[]; frame_danger=False; danger_side="center"; observed=[]
            depth_pairs = getattr(main, "depth_pairs", None)
            if depth_pairs is None:
                depth_pairs = deque(maxlen=240)
                main.depth_pairs = depth_pairs
            detections = []
            if res.boxes is not None and res.boxes.id is not None:
                ids=res.boxes.id.int().cpu().tolist(); xyxy=res.boxes.xyxy.cpu().numpy(); cls=res.boxes.cls.int().cpu().tolist()
                detections.extend((int(tid), box, str(names[c]).lower().replace(" ", "-")) for tid, box, c in zip(ids, xyxy, cls))
            # Merge fresh local auto-rickshaw predictions with the COCO detector.
            auto_ts, auto_predictions = auto_detector.get()
            if args.auto_model and now-auto_ts < .75:
                for box, label, confidence in auto_predictions:
                    if confidence < args.conf: continue
                    x1a,y1a,x2a,y2a=box; center=((x1a+x2a)/2,(y1a+y2a)/2)
                    # When COCO called the same vehicle a car, replace its label instead of
                    # drawing/announcing a duplicate box.
                    overlap_idx = None; best_iou = 0.0
                    for index, (_, other_box, other_label) in enumerate(detections):
                        if other_label not in ("car", "auto-rickshaw", "autorickshaw", "rickshaw"): continue
                        ox1,oy1,ox2,oy2=map(float,other_box)
                        iw=max(0,min(x2a,ox2)-max(x1a,ox1)); ih=max(0,min(y2a,oy2)-max(y1a,oy1))
                        inter=iw*ih; union=max(1,(x2a-x1a)*(y2a-y1a)+(ox2-ox1)*(oy2-oy1)-inter)
                        iou=inter/union
                        if iou>best_iou: best_iou,overlap_idx=iou,index
                    if overlap_idx is not None and best_iou>=.35:
                        base_tid,_,_=detections[overlap_idx]
                        detections[overlap_idx]=(base_tid,np.asarray(box,dtype=float),label)
                        continue
                    # Assign stable negative IDs by nearest recent auto box; COCO uses nonnegative IDs.
                    candidates=[(math.hypot(center[0]-(sum(t.bbox[::2])/2), center[1]-(sum(t.bbox[1::2])/2)), tid, t)
                                for tid,t in tracks.items() if tid < 0 and t.label==label and now-t.last_seen<1.0]
                    match=min(candidates,key=lambda item:item[0]) if candidates else None
                    if match and match[0] < max(60.0, .15*math.hypot(w,h)):
                        auto_tid=match[1]
                    else:
                        auto_tid=min([tid for tid in tracks if tid<0],default=0)-1
                    detections.append((auto_tid, np.asarray(box,dtype=float), label))
            for tid, box, label in detections:
                    x1,y1,x2,y2=map(float,box)
                    # Normalize label variants and announce all known COCO classes by default.
                    if label == "motorbike": label="motorcycle"
                    if label == "traffic-light": label="traffic light"
                    detected_boxes.append((x1,y1,x2,y2)); tr=tracks.get(tid) or tracks.setdefault(tid,Track(tid,label)); tr.label=label; tr.bbox=(x1,y1,x2,y2); tr.last_seen=now
                    height=KNOWN_HEIGHT_M.get(label, DEFAULT_HEIGHT_M); box_h=max(y2-y1,4)
                    z_box=float(np.clip(f_px*height/box_h,.3,60))
                    z=z_box
                    if depth is not None:
                        xa,xb=max(0,int(x1)),min(w,int(x2)); ya,yb=max(0,int(y1)),min(h,int(y2))
                        if xb>xa and yb>ya:
                            rel=float(np.median(depth[ya:yb,xa:xb])) # normalized inverse-depth proxy, near-high
                            depth_pairs.append((rel,z_box))
                            # Align this camera's relative depth scale to box-derived ranges.
                            # This is calibration, not metric ground truth; require several samples.
                            if len(depth_pairs) >= 8:
                                pairs=np.asarray(depth_pairs,dtype=np.float32)
                                slope, intercept=np.polyfit(pairs[:,0],pairs[:,1],1)
                                z_depth=float(np.clip(intercept+slope*rel,.3,60.0))
                                z=float((1-DEPTH_FUSE_WEIGHT)*z_box+DEPTH_FUSE_WEIGHT*z_depth)
                    tr.z_smooth=z if tr.z_smooth is None else EMA_ALPHA*z+(1-EMA_ALPHA)*tr.z_smooth
                    cx=(x1+x2)/2; x_m=(cx-w/2)/f_px*tr.z_smooth
                    tr.hist.append((now,tr.z_smooth,x_m)); update_motion(tr,now,own_speed)
                    level,will_hit=assess(tr)
                    if level in ("warning","danger") and tr.alert_started is None: tr.alert_started=now
                    if level == "none": tr.alert_started=None
                    if level != "none":
                        rank={"none":0,"info":1,"warning":2,"danger":3}
                        escalated=rank[level]>rank[tr.last_level]; due=now-tr.last_alert.get(level,0)>COOLDOWN[level]
                        if escalated or due:
                            latency=max(0,now-(tr.alert_started or now))
                            speaker.say(build_message(tr,level,will_hit),level); tr.last_alert[level]=now; tr.announced=True
                            if level in ("warning","danger"): eval_lats.append(latency)
                    tr.last_level=level
                    if level=="danger":
                        frame_danger=True; danger_side="left" if x_m < -.45 else "right" if x_m > .45 else "center"
                        if describer is not None: describer.cancel()
                    if level in ("warning","danger"):
                        beeper.cue(float(np.clip(x_m/max(tr.z_smooth,.1),-1,1)),tr.ttc,level=="danger")
                    if args.traffic_light and label=="traffic light":
                        col=traffic_color(frame,box)
                        if col and now-tr.last_alert.get("traffic",0)>4:
                            speaker.say(f"Traffic light appears {col}.","warning" if col=="red" else "info"); tr.last_alert["traffic"]=now
                    if args.find and label==args.find.lower().replace(" ","-") and now-find_last>2:
                        speaker.say(f"{args.find} {side_word(x_m,tr.z_smooth)}, about {tr.z_smooth:.0f} meters.","info"); find_last=now
                    observed.append(tr)
                    if writer:
                        writer.writerow([datetime.now().isoformat(timespec="milliseconds"),frame_no,tid,label,f"{tr.z_smooth:.2f}","" if tr.ttc is None else f"{tr.ttc:.2f}",level,"" if tr.alert_started is None else f"{max(0,now-tr.alert_started):.3f}"])
                    if label == "person":
                        tr.activity=activity(tr)
                        tr.direction=("toward" if tr.closing>.5 else "away" if tr.closing<-.5 else
                                      "crossing left-to-right" if tr.lateral_v>.35 else
                                      "crossing right-to-left" if tr.lateral_v<-.35 else "standing")
                    color={"danger":(0,0,255),"warning":(0,165,255),"info":(0,220,255),"none":(0,190,0)}[level]
                    if not args.no_display:
                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                        tag=f"{label} #{tid} ~{tr.z_smooth:.1f}m"+(f" TTC {tr.ttc:.1f}s" if tr.ttc is not None else "")
                        if tr.closing>MIN_CLOSING_SPEED: tag+=f" APPROACH {tr.closing:.1f}m/s"
                        elif tr.closing<-.6: tag+=f" AWAY {-tr.closing:.1f}m/s"
                        elif abs(tr.lateral_v)>.5: tag+=" CROSSING"
                        if label=="person": tag+=f" {getattr(tr,'activity','')} {getattr(tr,'direction','')} {math.hypot(tr.closing,tr.lateral_v):.1f}m/s"
                        cv2.putText(frame,tag,(int(x1),max(18,int(y1)-7)),cv2.FONT_HERSHEY_SIMPLEX,.52,color,2)

            # Small-object detections are asynchronous and informational; announce each class
            # once per session, except a requested find target which may be repeated on demand.
            prev_detection_boxes=detected_boxes.copy()
            small_ts, small_predictions=small_detector.get()
            pose_ts, pose_cues=pose_detector.get() if pose_detector is not None else (0.0,[])
            pose_context=[]
            if now-pose_ts<1.5:
                for pose_box,pose_text in pose_cues:
                    px1,py1,px2,py2=map(int,pose_box)
                    if display_enabled: cv2.putText(frame,pose_text,(px1,max(18,py1-24)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),2)
                    pose_context.append(pose_text)
            small_context=[]
            if now-small_ts<1.5:
                for box,label,conf in small_predictions:
                    x1s,y1s,x2s,y2s=box; direction=side_word(((x1s+x2s)/2-w/2)/max(w/2,1),1)
                    # side_word takes metric lateral position; use normalized horizontal position directly.
                    direction="on your left" if (x1s+x2s)/2<w*.42 else "on your right" if (x1s+x2s)/2>w*.58 else "ahead"
                    small_context.append(f"{label} {direction}")
                    if label.lower() not in small_detector.seen:
                        speaker.say(f"{label} {direction}.","info"); small_detector.seen.add(label.lower())
                    if display_enabled:
                        cv2.rectangle(frame,(int(x1s),int(y1s)),(int(x2s),int(y2s)),(255,180,0),2)
                        cv2.putText(frame,f"{label} {conf:.2f}",(int(x1s),max(18,int(y1s)-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,180,0),2)
            # Authoritative local measurements accompany optional cloud calls.
            def motion_description(t):
                if t.closing>MIN_CLOSING_SPEED: motion=f"approaching at {t.closing:.1f} m/s"
                elif t.closing<-.6: motion=f"moving away at {-t.closing:.1f} m/s"
                elif t.lateral_v>.5: motion=f"crossing left-to-right at {t.lateral_v:.1f} m/s"
                elif t.lateral_v<-.5: motion=f"crossing right-to-left at {-t.lateral_v:.1f} m/s"
                else: motion="no clear relative motion"
                item=f"{t.label} #{t.tid}, {side_word(t.x,t.z_smooth)}, {t.z_smooth:.1f} m, {motion}"
                if t.ttc is not None: item+=f", TTC {t.ttc:.1f} s"
                if t.label=="person": item+=f", person activity {getattr(t,'activity','unknown')}, direction {getattr(t,'direction','unknown')}"
                return item
            ranked_tracks=sorted(observed,key=lambda t:(t.ttc is None,t.ttc if t.ttc is not None else t.z_smooth))
            context_objects=[motion_description(t) for t in ranked_tracks[:8]]
            latest_context=f"User motion estimate from background optical flow: {motion_state}, approximately {own_speed:.1f} m/s. Tracked objects (prioritized by TTC): "+("; ".join(context_objects+small_context) or "none")
            if pose_context: latest_context += "; person pose cues: "+", ".join(pose_context[:3])
            audio_event=audio_monitor.get()
            if audio_event.get("label"):
                arrow="<-" if "left" in audio_event["direction"] else "->" if "right" in audio_event["direction"] else "^" if audio_event["direction"]=="ahead" else "v" if audio_event["direction"]=="behind you" else "?"
                sound_summary=(f"{arrow} {audio_event['label']} {audio_event['confidence']:.2f}, {audio_event['direction']}, "
                               f"{audio_event['trend']}, {audio_event['db']:.0f} dBFS, {audio_event['distance']}")
                evt_time=float(audio_event.get("time",0)); capture_mono=float(audio_event.get("captured_monotonic",time.monotonic()))
                # Match sounds to visible semantic classes and side where available.
                sound=audio_event["label"].lower()
                groups={"car horn":{"car","bus","truck","motorcycle","auto-rickshaw","autorickshaw","rickshaw"},
                        "siren":{"car","bus","truck","motorcycle","auto-rickshaw","autorickshaw","rickshaw"},
                        "vehicle engine":{"car","bus","truck","motorcycle","auto-rickshaw","autorickshaw","rickshaw"},
                        "motorcycle":{"motorcycle"},"truck":{"truck"},"bus":{"bus"},"train":{"train"},"dog bark":{"dog"}}
                wanted=groups.get(sound,set())
                def side_bucket(word): return "left" if "left" in word else "right" if "right" in word else "center"
                audio_side=side_bucket(audio_event["direction"])
                visual=next((t for t in observed if t.label in wanted and (audio_side=="center" or side_bucket(side_word(t.x,t.z_smooth))==audio_side)),None)
                fused=visual is not None
                angle_map={"on your left":-90,"left":-90,"front-left":-45,"ahead":0,"front-right":45,"right":90,"on your right":90,"behind you":180}
                angle=angle_map.get(audio_event["direction"],"")
                track_id=visual.tid if visual else ""; visual_class=visual.label if visual else ""
                visual_distance=f"{visual.z_smooth:.2f}" if visual else ""
                visual_ttc=f"{visual.ttc:.2f}" if visual and visual.ttc is not None else ""
                if audio_writer and evt_time>last_audio_log_time:
                    audio_writer.writerow([datetime.now().isoformat(timespec="milliseconds"),track_id,visual_class,audio_event["label"],f"{audio_event['confidence']:.3f}",audio_event["direction"],angle,audio_event["trend"],audio_event["distance"],visual_distance,visual_ttc,f"{audio_event['db']:.1f}",fused,"",f"{max(0,time.monotonic()-capture_mono):.3f}"])
                    last_audio_log_time=evt_time
                if evt_time and (audio_event["label"]!=last_audio_label or now-last_audio_spoken_time>4.0):
                    latency=max(0,time.monotonic()-capture_mono)
                    severe=sound in ("car horn","siren","tire squeal") and (sound=="tire squeal" or audio_event["trend"]=="getting louder")
                    level="danger" if severe else "warning"
                    confidence=min(1.0,audio_event["confidence"]+(.12 if fused else 0))
                    direction=audio_event["direction"]
                    if fused:
                        msg=f"{audio_event['label']} matches {visual.label} {side_word(visual.x,visual.z_smooth)}, {audio_event['trend']}, {audio_event['distance']}"
                    else:
                        msg=f"{audio_event['label']} {direction}, {audio_event['trend']}, {audio_event['distance']}"
                    if severe: msg="Danger! "+msg
                    speaker.say(msg,level)
                    eval_lats.append(latency)
                    if severe:
                        frame_danger=True; danger_side=audio_side
                    pan=-.8 if "left" in direction else .8 if "right" in direction else 0.0
                    beeper.cue(pan,2.0 if severe else 7.0,severe)
                    if severe and describer is not None: describer.cancel()
                    if audio_writer:
                        audio_writer.writerow([datetime.now().isoformat(timespec="milliseconds"),track_id,visual_class,audio_event["label"],f"{confidence:.3f}",direction,angle,audio_event["trend"],audio_event["distance"],visual_distance,visual_ttc,f"{audio_event['db']:.1f}",fused,level,f"{latency:.3f}"])
                    last_audio_label=audio_event["label"]; last_audio_spoken_time=now
                    latest_context += f"; sound: {sound_summary}"
            elif args.no_audio:
                sound_summary="Audio disabled"
            elif audio_event.get("error"):
                sound_summary="Mic/audio unavailable"
            else: sound_summary="No sound event"
            if describer is not None:
                result=describer.pop()
                if result: speaker.say(result[1],"info")
                if args.auto_describe and now-last_auto_description>=args.auto_describe:
                    describer.submit(frame,latest_context,"describe"); last_auto_description=now

            # Depth-only central path obstacle cue; relative depth is not metric calibrated, so warn conservatively.
            if depth is not None:
                xa,xb=int(w*.40),int(w*.60); ya,yb=int(h*.45),int(h*.95)
                strip=depth[ya:yb,xa:xb]; near=float(np.percentile(strip,85)) if strip.size else 0
                depth_distance=60.0
                if len(depth_pairs)>=8:
                    pairs=np.asarray(depth_pairs,dtype=np.float32); slope,intercept=np.polyfit(pairs[:,0],pairs[:,1],1)
                    depth_distance=float(np.clip(intercept+slope*near,.3,60.0))
                # If labels do not explain a very near central depth region, flag an unlabelled obstacle.
                labeled_center=any((x1 < xb and x2 > xa and y2 > ya) for x1,y1,x2,y2 in detected_boxes)
                if depth_distance<OBSTACLE_NEAR_M and not labeled_center and now-obstacle_last>3.0:
                    speaker.say("Obstacle ahead, stop and check your path.","warning"); obstacle_last=now
                    if not args.no_display: cv2.putText(frame,"OBSTACLE AHEAD",(w//2-150,h//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            for tid in [tid for tid,tr in tracks.items() if now-tr.last_seen>3]: del tracks[tid]
            if args.summary_every and now-last_summary>args.summary_every: summary(); last_summary=now
            dt=now-last_t; last_t=now; fps_s=.9*fps_s+.1*(1/max(dt,1e-6))
            if display_enabled:
                half=int(CORRIDOR_HALF_WIDTH_M/3*f_px)
                cv2.line(frame,(w//2-half,h),(w//2-half//3,h//2),(255,255,255),1); cv2.line(frame,(w//2+half,h),(w//2+half//3,h//2),(255,255,255),1)
                cv2.rectangle(frame,(8,8),(455,135),(20,20,20),-1)
                cv2.putText(frame,f"{fps_s:.1f} FPS  |  {motion_state} {own_speed:.1f} m/s",(18,32),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                cv2.putText(frame,f"RISK: {danger_side.upper() if frame_danger else 'CLEAR'}",(18,59),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255) if frame_danger else (0,220,0),2)
                cv2.putText(frame,f"SOUND: {sound_summary[:48]}",(18,87),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
                cv2.putText(frame,f"Mic {audio_event.get('db',-100):.0f} dBFS | floor {audio_event.get('noise_floor_db',-100):.0f} dBFS",(18,111),cv2.FONT_HERSHEY_SIMPLEX,.42,(210,210,210),1)
                if frame_danger: cv2.rectangle(frame,(w//2-105,8),(w//2+105,55),(0,0,220),-1); cv2.putText(frame,"DANGER",(w//2-90,43),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)
                try:
                    cv2.imshow("Blind Assist",frame)
                    key=cv2.waitKey(1)&0xff
                    if key==ord("q"): break
                    if key==ord("s"):
                        summary()
                        if describer is not None: describer.submit(frame,latest_context,"describe")
                    if key==ord("r") and args.ocr and ocr_reader:
                        try:
                            texts=ocr_reader.readtext(frame,detail=0,paragraph=True)
                            speaker.say("Text: "+"; ".join(texts[:5]) if texts else "No readable text found.","info")
                        except Exception as e: print(f"[OCR failed: {e}]")
                    if key==ord("f") and describer is not None:
                        target=find_target
                        if not target:
                            def request_find():
                                nonlocal find_target
                                try:
                                    name=input("Find which object? ").strip()
                                    if name:
                                        find_target=name
                                        describer.submit(frame.copy(),latest_context,"find",name)
                                except Exception: pass
                            threading.Thread(target=request_find,daemon=True).start()
                        elif now-find_last>1.0:
                            describer.submit(frame,latest_context,"find",target); find_last=now
                    if key==ord("r") and describer is not None and not (args.ocr and ocr_reader):
                        describer.submit(frame,latest_context,"read")
                except cv2.error as e:
                    display_enabled=False
                    print(f"[OpenCV display unavailable; continuing without video window: {e}]")
            if args.record:
                if video is None:
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v"); video=cv2.VideoWriter(args.record,fourcc,20.0,(w,h))
                video.write(frame)
            if args.evaluate and eval_index[0]<len(eval_targets) and observed and not eval_active[0]:
                target=eval_targets[eval_index[0]]
                snapshot=[(t.tid,t.z_smooth) for t in observed]
                eval_active[0]=True
                def evaluation_prompt(target_m=target, snapshot_rows=snapshot):
                    try:
                        answer=input(f"Place an object at {target_m:g} m; enter visible track ID or Enter for nearest: ").strip()
                        chosen=next((row for row in snapshot_rows if str(row[0])==answer),None) if answer else min(snapshot_rows,key=lambda row:row[1])
                        if chosen:
                            error=abs(chosen[1]-target_m); eval_errors.append(error)
                            print(f"Estimated {chosen[1]:.2f} m; absolute error {error:.2f} m")
                    except Exception as e: print(f"[evaluation prompt unavailable: {e}]")
                    finally: eval_index[0]+=1; eval_active[0]=False
                threading.Thread(target=evaluation_prompt,daemon=True).start()
    finally:
        reader.release()
        try: cv2.destroyAllWindows()
        except cv2.error: pass
        if video: video.release()
        if csv_file: csv_file.close()
        if audio_csv: audio_csv.close()
        audio_monitor.stop()
        if args.evaluate:
            print(f"Average distance error: {np.mean(eval_errors):.2f} m" if eval_errors else "Average distance error: no calibration samples")
            print(f"Average FPS: {fps_s:.2f}")
            print(f"Average alert latency: {np.mean(eval_lats)*1000:.0f} ms" if eval_lats else "Average alert latency: no alerts")


if __name__ == "__main__":
    main()
