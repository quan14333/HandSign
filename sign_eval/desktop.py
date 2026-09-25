"""Desktop selection, responsive processing, and evaluation result windows.

Only the main thread touches Tk. The worker returns data through a queue;
no terminal parsing or previous evaluation file is used to populate the UI.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
import unicodedata
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable, Literal

from .evaluator import canonical_label


REGION_NAMES = {"left_hand": "Tay trái", "right_hand": "Tay phải", "face": "Khuôn mặt"}
STATUS_NAMES = {
    "correct": "Đúng từ mục tiêu",
    "right_label_needs_practice": "Đúng từ — cần luyện thêm động tác",
    "incorrect_label": "Khác từ mục tiêu — hãy thực hiện lại",
    "needs_label_confirmation": "Chưa nhận diện được từ — hãy thực hiện lại",
    "not_scored": "Chưa đủ dữ liệu để chấm điểm",
}
FORM_NAMES = {
    "excellent": "Rất tốt", "good": "Tốt", "needs_practice": "Cần luyện thêm",
    "far_from_reference": "Còn khác nhiều so với mẫu", "not_scored": "Chưa chấm điểm",
}


def _number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if percent else f"{value:.2f}"


def result_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Format existing scores without recalculating or filling missing scores."""
    form = payload.get("form", {})
    recognition = payload.get("recognition", {})
    tracking = payload.get("tracking", {})
    score = form.get("score")
    region_rows = []
    if score is not None:
        for item in payload.get("feedback", []):
            if item.get("region") not in REGION_NAMES or "score" not in item:
                continue
            region_rows.append((
                REGION_NAMES[item["region"]], _number(item["score"]),
                _number(item.get("weight"), percent=True), _number(item.get("distance")),
                _number(item.get("threshold")),
                "Khớp mẫu" if item.get("severity") == "ok" else "Cần chú ý",
            ))
    diagnostics = [
        f"Mức độ động tác: {FORM_NAMES.get(form.get('status'), 'Chưa chấm điểm')}",
        f"Độ tin cậy nhận diện: {_number(recognition.get('confidence'), percent=True)} (không phải điểm động tác)",
        f"Chênh lệch hai dự đoán đầu: {_number(recognition.get('margin'), percent=True)}",
        f"Theo dõi tay trái: {_number(tracking.get('left_hand_detected_fraction'), percent=True)}",
        f"Theo dõi tay phải: {_number(tracking.get('right_hand_detected_fraction'), percent=True)}",
        f"Theo dõi khuôn mặt: {_number(tracking.get('face_detected_fraction'), percent=True)}",
        f"Số khung hình: {tracking.get('frame_count', '—')}",
        f"Số mẫu tham chiếu: {form.get('reference_count', '—')}",
        f"Khoảng cách DTW: {_number(form.get('comparison_distance'))}",
        f"Phân vị tham chiếu: {_number(form.get('reference_percentile'))}",
    ]
    messages = [item["message"] for item in payload.get("feedback", []) if item.get("message")]
    return {
        "title": STATUS_NAMES.get(payload.get("status"), "Kết quả đánh giá"),
        "score": "Chưa chấm điểm" if score is None else f"{score:.2f} / 100",
        "target": payload.get("target_label", "—"),
        "prediction": recognition.get("predicted_label") or "Chưa nhận diện được",
        "rows": region_rows,
        "feedback": "\n\n".join(messages) or "Chưa có góp ý.",
        "diagnostics": "\n".join(diagnostics),
    }


def _window(title: str, width: int, height: int) -> tk.Tk:
    root = tk.Tk()
    root.title(title)
    root.configure(background="#f4f7fa")
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.minsize(min(width, 640), min(height, 480))
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#f4f7fa")
    style.configure("TLabel", background="#f4f7fa", font=("Segoe UI", 11))
    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#173d50")
    style.configure("Score.TLabel", font=("Segoe UI", 30, "bold"), foreground="#087e80")
    style.configure("TButton", font=("Segoe UI", 11), padding=(14, 8))
    style.configure("Treeview", font=("Segoe UI", 10), rowheight=30)
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
    return root


def show_notice(title: str, message: str, *, error: bool = False) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        show = messagebox.showerror if error else messagebox.showinfo
        show(title, message, parent=root)
    finally:
        root.destroy()


def suggest_labels(labels: list[str], query: str) -> list[str]:
    """Prefix search accepts Vietnamese text with or without diacritics."""
    def key(value: str) -> str:
        text = unicodedata.normalize("NFD", canonical_label(value)).replace("đ", "d")
        return "".join(char for char in text if not unicodedata.combining(char))
    prefix = key(query)
    return [label for label in labels if key(label).startswith(prefix)]


class LabelSelector(ttk.Frame):
    """An always-visible suggestion list; typing never requires opening a dropdown."""

    def __init__(self, parent: tk.Misc, labels: list[str]) -> None:
        super().__init__(parent)
        self.labels = labels
        self.value = tk.StringVar(self)
        self.entry = ttk.Entry(self, textvariable=self.value, font=("Segoe UI", 13))
        self.entry.pack(fill="x")
        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.suggestions = tk.Listbox(list_frame, height=8, font=("Segoe UI", 12),
                                      exportselection=False, activestyle="none")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.suggestions.yview)
        self.suggestions.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.suggestions.pack(fill="both", expand=True)
        self.hint = ttk.Label(self)
        self.hint.pack(anchor="w", pady=6)
        self.value.trace_add("write", self._refresh)
        self.entry.bind("<Down>", self._focus_list)
        self.suggestions.bind("<ButtonRelease-1>", self._choose)
        self.suggestions.bind("<Return>", self._choose)
        self._refresh()

    def _refresh(self, *_args: Any) -> None:
        self.matches = suggest_labels(self.labels, self.value.get())
        self.suggestions.delete(0, "end")
        for label in self.matches:
            self.suggestions.insert("end", label)
        self.hint.configure(text=f"{len(self.matches)} từ gợi ý — bấm để điền vào ô nhập"
                            if self.matches else "Không có từ phù hợp. Hãy thử nhập chữ khác.")

    def _focus_list(self, _event: Any) -> str:
        if self.matches:
            self.suggestions.focus_set()
            self.suggestions.selection_set(0)
            self.suggestions.activate(0)
        return "break"

    def _choose(self, _event: Any) -> str:
        selected = self.suggestions.curselection()
        if selected:
            label = self.suggestions.get(selected[0])
            self.value.set(label)
            self.entry.focus_set()
            self.entry.icursor("end")
        return "break"

    def selected_label(self) -> str | None:
        exact = {canonical_label(label): label for label in self.labels}
        return exact.get(canonical_label(self.value.get()))


def choose_target_label(labels: list[str]) -> str | None:
    if not labels:
        raise ValueError("Chưa có từ mục tiêu trong dữ liệu tham chiếu.")
    root = _window("HandSign — Chọn từ luyện tập", 640, 495)
    content = ttk.Frame(root, padding=24)
    content.pack(fill="both", expand=True)
    ttk.Label(content, text="Bạn muốn luyện từ nào?", style="Title.TLabel").pack(anchor="w")
    ttk.Label(content, text="Nhập B, Bế hoặc be để tìm từ, rồi chọn gợi ý.").pack(anchor="w", pady=(10, 16))
    selection = LabelSelector(content, labels)
    selection.pack(fill="both", expand=True)
    selection.entry.focus_set()
    selected: list[str] = []

    def accept() -> None:
        value = selection.selected_label()
        if value is None:
            messagebox.showinfo("Chọn từ mục tiêu", "Hãy chọn một từ trong danh sách.", parent=root)
            return
        selected.append(value)
        root.destroy()

    root.bind("<Return>", lambda _event: accept())
    ttk.Button(content, text="Xem động tác mẫu", command=accept).pack(anchor="e", pady=(12, 0))
    root.mainloop()
    return selected[0] if selected else None


SampleWindowAction = bool | Literal["choose_another"]


def show_sample_window(target_label: str, video_root: Path | None = None) -> SampleWindowAction:
    """Show a demonstration and return the user's next navigation action."""
    from .sample_player import SamplePlayer

    root = _window("HandSign — Động tác mẫu", 900, 720)
    content = ttk.Frame(root, padding=24)
    content.pack(fill="both", expand=True)
    ttk.Label(content, text="Động tác mẫu", style="Title.TLabel", anchor="center").pack(fill="x")
    ttk.Label(content, text=target_label, anchor="center", font=("Segoe UI", 15)).pack(fill="x", pady=(6, 18))
    action: SampleWindowAction = False

    def start_practice() -> None:
        nonlocal action
        action = True
        root.destroy()

    def choose_another() -> None:
        nonlocal action
        action = "choose_another"
        root.destroy()

    actions = ttk.Frame(content)
    actions.pack(side="bottom", fill="x", pady=(16, 0))
    ttk.Button(actions, text="Luyện tập với AI", command=start_practice).pack(side="right")
    ttk.Button(actions, text="Đóng", command=root.destroy).pack(side="left")
    ttk.Button(actions, text="Chọn từ khác", command=choose_another).pack(side="left", padx=12)
    player = SamplePlayer(content, target_label, video_root)
    player.pack(fill="both", expand=True)
    root.mainloop()
    return action


def _text_panel(parent: tk.Misc, text: str) -> None:
    widget = scrolledtext.ScrolledText(
        parent, wrap="word", font=("Segoe UI", 11), relief="flat", padx=12, pady=12,
        background="white", foreground="#243746",
    )
    widget.insert("1.0", text)
    widget.configure(state="disabled")
    widget.pack(fill="both", expand=True)


def render_result(parent: tk.Misc, payload: dict[str, Any]) -> None:
    view = result_view(payload)
    ttk.Label(parent, text=view["title"], style="Title.TLabel", wraplength=800).pack(anchor="w")
    ttk.Label(parent, text=view["score"], style="Score.TLabel").pack(anchor="w", pady=(12, 8))
    ttk.Label(parent, text=f"Từ mục tiêu: {view['target']}   •   AI nhận diện: {view['prediction']}",
              wraplength=800).pack(anchor="w", pady=(0, 16))
    if view["rows"]:
        columns = ("region", "score", "weight", "distance", "threshold", "status")
        table = ttk.Treeview(parent, columns=columns, show="headings", height=len(view["rows"]))
        for key, title, width in zip(columns,
                ("Vùng", "Điểm / 100", "Trọng số", "Độ lệch", "Ngưỡng", "Nhận xét"),
                (130, 100, 100, 100, 100, 140)):
            table.heading(key, text=title)
            table.column(key, width=width, minwidth=70, anchor="center")
        for row in view["rows"]:
            table.insert("", "end", values=row)
        table.pack(fill="x", pady=(0, 16))
    tabs = ttk.Notebook(parent)
    tabs.pack(fill="both", expand=True)
    for title, text in [("Góp ý", view["feedback"]), ("Thông số chi tiết", view["diagnostics"])]:
        tab = ttk.Frame(tabs)
        tabs.add(tab, text=title)
        _text_panel(tab, text)


def _run_task(task: Callable[[], dict[str, Any]], messages: queue.Queue) -> None:
    try:
        messages.put(("result", task()))
    except Exception as error:
        messages.put(("error", f"{type(error).__name__}: {error}"))


def show_evaluation_window(task: Callable[[], dict[str, Any]], target_label: str) -> bool:
    """Show progress then results; return True only for an explicit re-record."""
    root = _window("HandSign — AI đang xử lí", 900, 700)
    content = ttk.Frame(root, padding=24)
    content.pack(fill="both", expand=True)
    ttk.Label(content, text="AI đang xử lí…", style="Title.TLabel").pack(anchor="w", pady=(60, 16))
    ttk.Label(content, text=f"Đang nhận diện và đánh giá từ: {target_label}").pack(anchor="w")
    ttk.Label(content, text="Lần chạy đầu có thể lâu hơn do cần tải model.").pack(anchor="w", pady=10)
    progress = ttk.Progressbar(content, mode="indeterminate")
    progress.pack(fill="x", pady=16)
    progress.start(15)
    elapsed = ttk.Label(content, text="Đã xử lí: 0 giây")
    elapsed.pack(anchor="w")
    messages: queue.Queue = queue.Queue()
    started_at = time.monotonic()
    retry = False

    def repeat() -> None:
        nonlocal retry
        retry = True
        root.destroy()

    def poll() -> None:
        try:
            kind, data = messages.get_nowait()
        except queue.Empty:
            elapsed.configure(text=f"Đã xử lí: {int(time.monotonic() - started_at)} giây")
            root.after(100, poll)
            return
        progress.stop()
        for child in content.winfo_children():
            child.destroy()
        root.title("HandSign — Kết quả" if kind == "result" else "HandSign — Không thể đánh giá")
        buttons = ttk.Frame(content)
        buttons.pack(side="bottom", fill="x", pady=(16, 0))
        ttk.Button(buttons, text="Đóng", command=root.destroy).pack(side="right")
        ttk.Button(buttons, text="Quay lại động tác", command=repeat).pack(side="right", padx=12)
        if kind == "result":
            render_result(content, data)
        else:
            ttk.Label(content, text="Không thể hoàn tất đánh giá", style="Title.TLabel").pack(anchor="w", pady=12)
            _text_panel(content, "Lần này chưa có điểm.\n\n" + data)

    def start() -> None:
        threading.Thread(target=_run_task, args=(task, messages), daemon=True).start()
        poll()

    root.after(100, start)
    root.mainloop()
    return retry
