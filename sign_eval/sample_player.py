"""Optional silent video playback embedded in the desktop lesson window."""

import queue
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from .samples import SampleVideo, open_sample_video


class SamplePlayer(ttk.Frame):
    def __init__(self, parent: tk.Misc, label: str, video_root: Path | None = None) -> None:
        super().__init__(parent)
        self.video: SampleVideo | None = None
        self.playing = False
        self.photo = None
        self._timer = None
        self._poll_timer = None
        self._closed = False
        self._lock = threading.Lock()
        self._loaded: queue.Queue = queue.Queue()
        self.screen = tk.Label(self, text="Đang tìm video mẫu…", bg="#142c3a", fg="white",
                               font=("Segoe UI", 16), anchor="center")
        self.screen.pack(fill="both", expand=True)
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(10, 0))
        self.pause = ttk.Button(controls, text="Tạm dừng", command=self.toggle, state="disabled")
        self.pause.pack(side="left")
        self.replay = ttk.Button(controls, text="Xem lại", command=self.restart, state="disabled")
        self.replay.pack(side="left", padx=8)
        self.status = ttk.Label(controls, text="Bạn có thể luyện tập mà không cần video mẫu.")
        self.status.pack(side="left", padx=6)
        self.bind("<Destroy>", self._destroyed)

        def load() -> None:
            try:
                video = open_sample_video(label, video_root)
            except Exception:
                # Optional media must never block or crash the practice flow.
                video = None
            with self._lock:
                if self._closed:
                    if video is not None:
                        video.close()
                else:
                    self._loaded.put(video)

        threading.Thread(target=load, daemon=True).start()
        self._poll_timer = self.after(60, self._poll)

    def _poll(self) -> None:
        try:
            self.video = self._loaded.get_nowait()
        except queue.Empty:
            self._poll_timer = self.after(60, self._poll)
            return
        self._poll_timer = None
        if self.video is None:
            self._unavailable()
            return
        self.pause.configure(state="normal")
        self.replay.configure(state="normal")
        self.status.configure(text="Quan sát động tác, rồi chọn Luyện tập với AI.")
        self.playing = True
        self._render(self.video.first_frame)
        self._schedule()

    def _render(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((max(1, self.screen.winfo_width()), max(1, self.screen.winfo_height())),
                        Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image, master=self.screen)
        self.screen.configure(image=self.photo, text="")

    def _schedule(self) -> None:
        if self.playing and self.video is not None:
            self._timer = self.after(max(10, round(1000 / self.video.fps)), self._tick)

    def _tick(self) -> None:
        self._timer = None
        if not self.playing or self.video is None:
            return
        try:
            ok, frame = self.video.capture.read()
            if not ok or frame is None:
                total = self.video.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                position = self.video.capture.get(cv2.CAP_PROP_POS_FRAMES)
                if total > 0 and position < total - 1:
                    self._unavailable()
                    return
                self.playing = False
                self.pause.configure(state="disabled")
                self.status.configure(text="Đã phát xong. Nhấn Xem lại để xem lần nữa.")
                return
            self._render(frame)
        except (cv2.error, OSError, ValueError):
            self._unavailable()
            return
        self._schedule()

    def toggle(self) -> None:
        if self.video is None:
            return
        self.playing = not self.playing
        self.pause.configure(text="Tạm dừng" if self.playing else "Tiếp tục")
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        self._schedule()

    def restart(self) -> None:
        if self.video is None:
            return
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        try:
            if not self.video.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                self._unavailable()
                return
        except cv2.error:
            self._unavailable()
            return
        self.playing = True
        self.pause.configure(state="normal", text="Tạm dừng")
        self._tick()

    def _unavailable(self) -> None:
        self.playing = False
        if self.video is not None:
            self.video.close()
            self.video = None
        self.screen.configure(image="", text="Hiện không có video")
        self.photo = None
        self.pause.configure(state="disabled")
        self.replay.configure(state="disabled")
        self.status.configure(text="Bạn vẫn có thể chọn Luyện tập với AI.")

    def _destroyed(self, event) -> None:
        if event.widget is not self:
            return
        with self._lock:
            self._closed = True
            try:
                pending = self._loaded.get_nowait()
            except queue.Empty:
                pending = None
            if pending is not None:
                pending.close()
        for timer in (self._timer, self._poll_timer):
            if timer is not None:
                self.after_cancel(timer)
        if self.video is not None:
            self.video.close()
            self.video = None
