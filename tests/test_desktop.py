import queue
import threading
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from sign_eval import desktop


def scored_payload(score=84.84):
    return {
        "status": "correct", "target_label": "Bế mạc",
        "recognition": {"predicted_label": "Bế mạc", "confidence": 0.4, "margin": 0.1},
        "form": {"score": score, "status": "good", "comparison_distance": 4.17},
        "feedback": [{"region": "right_hand", "score": 81.05, "weight": 0.8,
                      "distance": 0.9662, "threshold": 1.302055,
                      "severity": "ok", "message": "Tay phải khớp tốt."}],
    }


class ResultPresentationTests(unittest.TestCase):
    def test_displays_scores_and_model_confidence_separately(self):
        view = desktop.result_view(scored_payload())
        self.assertEqual(view["score"], "84.84 / 100")
        self.assertEqual(view["prediction"], "Bế mạc")
        self.assertEqual(view["rows"][0][:3], ("Tay phải", "81.05", "80.0%"))
        self.assertIn("40.0% (không phải điểm động tác)", view["diagnostics"])
        self.assertEqual(desktop.result_view(scored_payload(0))["score"], "0.00 / 100")

    def test_unscored_result_has_no_numeric_score_or_region_table(self):
        payload = scored_payload(None)
        payload["status"] = "incorrect_label"
        payload["recognition"]["predicted_label"] = "Cảm ơn"
        # Even unexpected regional data must not display scores when unscored.
        view = desktop.result_view(payload)
        self.assertEqual(view["score"], "Chưa chấm điểm")
        self.assertEqual(view["rows"], [])
        self.assertIn("thực hiện lại", view["title"])
        self.assertEqual(view["prediction"], "Cảm ơn")

    def test_worker_reports_errors_instead_of_stale_results(self):
        messages = queue.Queue()
        def fail():
            raise RuntimeError("Cannot load model")
        desktop._run_task(fail, messages)
        kind, data = messages.get_nowait()
        self.assertEqual(kind, "error")
        self.assertIn("Cannot load model", data)


class DesktopWindowTests(unittest.TestCase):
    def make_root(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        return root

    def test_live_suggestions_and_click_to_fill(self):
        root = self.make_root()
        try:
            picker = desktop.LabelSelector(root, ["Ban ngày", "Bế mạc", "Cá biển"])
            picker.value.set("B")
            self.assertEqual(picker.suggestions.get(0, "end"), ("Ban ngày", "Bế mạc"))
            picker.value.set("Bế")
            picker.suggestions.selection_set(0)
            picker._choose(None)
            self.assertEqual(picker.selected_label(), "Bế mạc")
        finally:
            root.update_idletasks()
            root.destroy()

    def test_playable_video_renders_and_releases_on_close(self):
        root = self.make_root()
        import numpy as np
        from sign_eval.sample_player import SamplePlayer
        from sign_eval.samples import SampleVideo
        capture = Mock()
        capture.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
        video = SampleVideo(capture, np.zeros((32, 32, 3), dtype=np.uint8), 25)
        with patch("sign_eval.sample_player.open_sample_video", return_value=video):
            player = SamplePlayer(root, "Bế mạc")
            player.pack(fill="both", expand=True)
            root.after(300, root.quit)
            root.mainloop()
            self.assertIsNotNone(player.photo)
            self.assertEqual(player.screen.cget("text"), "")
            self.assertEqual(str(player.pause.cget("state")), "normal")
            player.toggle()
            self.assertFalse(player.playing)
            player.restart()
            self.assertTrue(player.playing)
            root.destroy()
        capture.release.assert_called_once()

    def test_missing_video_still_allows_practice(self):
        root = self.make_root()
        from sign_eval.sample_player import SamplePlayer
        from tkinter import ttk
        seen = []
        def inspect_then_practice():
            def walk(widget):
                yield widget
                for child in widget.winfo_children():
                    yield from walk(child)
            for widget in walk(root):
                if isinstance(widget, SamplePlayer):
                    seen.append(widget.screen.cget("text"))
            for widget in walk(root):
                if isinstance(widget, ttk.Button) and widget.cget("text") == "Luyện tập với AI":
                    widget.invoke()
                    return
        root.after(400, inspect_then_practice)
        root.after(3000, root.destroy)
        with patch.object(desktop, "_window", return_value=root), \
                patch("sign_eval.sample_player.open_sample_video", return_value=None):
            self.assertTrue(desktop.show_sample_window("Bế mạc"))
        self.assertEqual(seen, ["Hiện không có video"])

    def test_sample_window_can_choose_another_label(self):
        root = self.make_root()
        from tkinter import ttk
        def choose_another():
            def walk(widget):
                yield widget
                for child in widget.winfo_children():
                    yield from walk(child)
            for widget in walk(root):
                if isinstance(widget, ttk.Button) and widget.cget("text") == "Chọn từ khác":
                    widget.invoke()
                    return
        root.after(100, choose_another)
        root.after(3000, root.destroy)
        with patch.object(desktop, "_window", return_value=root), \
                patch("sign_eval.sample_player.open_sample_video", return_value=None):
            self.assertEqual(desktop.show_sample_window("Bế mạc"), "choose_another")

    def test_real_tk_processing_window_stays_responsive_then_renders_result(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        self.addCleanup(lambda: root.destroy() if root.tk.call("winfo", "exists", ".") else None)
        task_started = threading.Event()
        release_task = threading.Event()
        worker_threads = []
        heartbeat = []
        rendered = []
        original_render = desktop.render_result

        def task():
            worker_threads.append(threading.get_ident())
            task_started.set()
            if not release_task.wait(3):
                raise RuntimeError("GUI heartbeat did not run")
            return scored_payload()

        def tick():
            if not task_started.is_set():
                root.after(20, tick)
                return
            heartbeat.append(True)
            release_task.set()

        def render(parent, payload):
            rendered.append(threading.get_ident())
            original_render(parent, payload)
            root.after(10, root.quit)

        root.after(20, tick)
        root.after(5000, root.quit)
        with patch.object(desktop, "_window", return_value=root), \
                patch.object(desktop, "render_result", side_effect=render):
            retry = desktop.show_evaluation_window(task, "Bế mạc")
        self.assertFalse(retry)
        self.assertTrue(heartbeat)
        self.assertEqual(rendered, [threading.get_ident()])
        self.assertNotEqual(worker_threads, [threading.get_ident()])


if __name__ == "__main__":
    unittest.main()
