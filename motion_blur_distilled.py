from __future__ import annotations
import argparse, os, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Deque, List, Tuple, Optional
import threading

import cv2
import numpy as np
from tqdm import tqdm

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ────────────────────────────────────────────────────────────────
DEFAULT_TEMPORAL_FRAMES = 7
DEFAULT_KSIZE           = 15
DEFAULT_BATCH_SIZE      = 32
DEFAULT_WORKERS         = min(os.cpu_count() or 4, 8)
QUEUE_MAX               = 4

# ────────────────────────────────────────────────────────────────
def _ensure_odd(n: int, name: str) -> int:
    n = max(3, int(n))
    if n % 2 == 0:
        n += 1
        print(f"[warn] {name} must be odd — bumped to {n}")
    return n

# ────────────────────────────────────────────────────────────────
def build_kernel(ksize: int, direction: str) -> Optional[np.ndarray]:
    if direction == "none": return None
    k = np.zeros((ksize, ksize), dtype=np.float32)
    mid = ksize // 2
    if direction == "horizontal":
        k[mid, :] = 1.0 / ksize
    elif direction == "vertical":
        k[:, mid] = 1.0 / ksize
    return k

def apply_directional(frame: np.ndarray, kernel: Optional[np.ndarray], direction: str) -> np.ndarray:
    if direction == "none": return frame
    if direction == "both":
        frame = cv2.filter2D(frame, -1, build_kernel(kernel.shape[0], "horizontal"))
        frame = cv2.filter2D(frame, -1, build_kernel(kernel.shape[0], "vertical"))
    else:
        frame = cv2.filter2D(frame, -1, kernel)
    return frame

# ────────────────────────────────────────────────────────────────
class FrameProducer:
    _SENTINEL = object()
    def __init__(self, path: str, prefetch: int = DEFAULT_WORKERS * 2):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise IOError(f"Cannot open video {path!r}")
        self.queue: Deque = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.maxlen = max(prefetch, 8)
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def fps(self): return self.cap.get(cv2.CAP_PROP_FPS) or 25.0
    @property
    def width(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    @property
    def height(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    @property
    def frame_count(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self): return self
    def __next__(self):
        while True:
            with self.cond:
                while not self.queue and not self.done: self.cond.wait(timeout=0.05)
                if self.queue:
                    item = self.queue.popleft()
                    self.cond.notify_all()
                    if item is self._SENTINEL: raise StopIteration
                    return item
                if self.done: raise StopIteration

    def _run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                with self.cond:
                    while len(self.queue) >= self.maxlen: self.cond.wait(timeout=0.05)
                    self.queue.append(frame)
                    self.cond.notify_all()
        finally:
            with self.cond:
                self.queue.append(self._SENTINEL)
                self.done = True
                self.cond.notify_all()
            self.cap.release()

# ────────────────────────────────────────────────────────────────
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def weighted_blend(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        H, W, C = stack.shape[1], stack.shape[2], stack.shape[3]
        out = np.zeros((H, W, C), dtype=np.float32)
        for i in prange(stack.shape[0]):
            for y in prange(H):
                for x in prange(W):
                    for c in prange(C):
                        out[y, x, c] += stack[i, y, x, c] * weights[i]
        return out
else:
    def weighted_blend(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.tensordot(weights, stack, axes=([0],[0]))

# ────────────────────────────────────────────────────────────────
class TemporalBlurEngine:
    def __init__(self, n_frames: int, direction: str = "none", ksize: int = DEFAULT_KSIZE, workers: int = DEFAULT_WORKERS):
        self.n_frames = n_frames
        self.direction = direction
        self.ksize = ksize
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.weights = self._compute_weights(n_frames)
        self.kernel = build_kernel(ksize, direction) if direction != "none" else None

    def _compute_weights(self, n: int) -> np.ndarray:
        half = n//2
        ramp = np.arange(1, half+2, dtype=np.float32)
        w = np.concatenate([ramp, ramp[-2::-1]])
        return w / w.sum()

    def process_batch(self, frames: List[np.ndarray], ring: Deque[np.ndarray]) -> List[np.ndarray]:
        results: List[np.ndarray] = [None]*len(frames)
        futures = []
        for i, f in enumerate(frames):
            ring.append(f.astype(np.float32, copy=False))
            snapshot = np.stack(ring, axis=0)
            w = self.weights if snapshot.shape[0]==len(self.weights) else self.weights[-snapshot.shape[0]:]
            w /= w.sum()
            futures.append(self.executor.submit(self._blend_and_blur, snapshot, w, i))
        for fut in futures:
            res, idx = fut.result()
            results[idx] = res
        return results

    def _blend_and_blur(self, stack: np.ndarray, weights: np.ndarray, idx: int) -> Tuple[np.ndarray,int]:
        blended = weighted_blend(stack, weights)
        frame = np.clip(blended, 0, 255).astype(np.uint8)
        if self.kernel is not None:
            frame = apply_directional(frame, self.kernel, self.direction)
        return frame, idx

    def shutdown(self): self.executor.shutdown(wait=True)

# ────────────────────────────────────────────────────────────────
class BatchWriter:
    def __init__(self, path:str, fps:float, width:int, height:int, codec:str="mp4v", batch_size:int=DEFAULT_BATCH_SIZE):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width,height))
        if not self.writer.isOpened(): raise IOError(f"Cannot open {path} with codec {codec}")
        self.batch_size = batch_size
        self.buffer = []
        self.queue: Deque[List[np.ndarray]] = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, frame: np.ndarray):
        self.buffer.append(frame)
        if len(self.buffer) >= self.batch_size: self._flush()

    def close(self):
        if self.buffer: self._flush()
        with self.cond: self.done=True; self.cond.notify_all()
        self.thread.join()
        self.writer.release()

    def _flush(self):
        batch = self.buffer[:]; self.buffer.clear()
        with self.cond:
            while len(self.queue)>=QUEUE_MAX: self.cond.wait(timeout=0.05)
            self.queue.append(batch)
            self.cond.notify_all()

    def _run(self):
        while True:
            with self.cond:
                while not self.queue and not self.done: self.cond.wait(timeout=0.05)
                if self.queue: batch=self.queue.popleft(); self.cond.notify_all()
                elif self.done: break
                else: continue
            for f in batch: self.writer.write(f)

# ────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace):
    input_path = Path(args.input)
    if not input_path.exists(): sys.exit(f"Input {input_path} not found")
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_blurred.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = _ensure_odd(args.frames,"--frames")
    ksize    = _ensure_odd(args.ksize,"--ksize")
    direction= args.dir.lower()
    batch_size=args.batch
    workers  = args.workers

    producer = FrameProducer(str(input_path), prefetch=workers*4)
    engine = TemporalBlurEngine(n_frames, direction, ksize, workers)
    writer = BatchWriter(str(output_path), producer.fps, producer.width, producer.height, args.codec, batch_size)

    ring: Deque = deque(maxlen=n_frames)
    frame_counter = 0
    raw_batch: List[np.ndarray] = []

    pbar = tqdm(total=producer.frame_count if producer.frame_count>0 else None, desc="Processing", unit="frame", colour="cyan")
    try:
        for frame in producer:
            raw_batch.append(frame)
            if len(raw_batch) >= batch_size:
                blurred = engine.process_batch(raw_batch, ring)
                for f in blurred: writer.write(f)
                frame_counter += len(raw_batch)
                pbar.update(len(raw_batch))
                raw_batch.clear()
        if raw_batch:
            blurred = engine.process_batch(raw_batch, ring)
            for f in blurred: writer.write(f)
            frame_counter += len(raw_batch)
            pbar.update(len(raw_batch))
    finally:
        pbar.close()
        writer.close()
        engine.shutdown()

    print(f"[done] Frames written: {frame_counter}, output saved to {output_path}")

# ────────────────────────────────────────────────────────────────
def build_parser(): 
    p=argparse.ArgumentParser(description="Optimized motion blur for MP4/MKV")
    p.add_argument("--input",required=True)
    p.add_argument("--output")
    p.add_argument("--frames",type=int,default=DEFAULT_TEMPORAL_FRAMES)
    p.add_argument("--dir",default="none",choices=["none","horizontal","vertical","both"])
    p.add_argument("--ksize",type=int,default=DEFAULT_KSIZE)
    p.add_argument("--batch",type=int,default=DEFAULT_BATCH_SIZE)
    p.add_argument("--workers",type=int,default=DEFAULT_WORKERS)
    p.add_argument("--codec",default="mp4v",choices=["mp4v","XVID"])
    return p

if __name__=="__main__":
    run_pipeline(build_parser().parse_args())
