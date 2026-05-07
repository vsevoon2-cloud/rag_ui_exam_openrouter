import ctypes
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import ImageGrab

from exam_assistant_core import (
    OpenRouterClient,
    Retriever,
    default_config,
    load_saved_settings,
    save_settings,
)


HOTKEY_ID = 1
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
VK_Q = 0x51


class ScreenSelector:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.overlay = tk.Toplevel(root)
        self.overlay.withdraw()
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-alpha", 0.25)
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(bg="black")
        self.canvas = tk.Canvas(self.overlay, cursor="cross", bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.start_x = 0
        self.start_y = 0
        self.rect = None
        self.result = None
        self.done = threading.Event()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.overlay.bind("<Escape>", self._on_cancel)

    def _on_press(self, event: tk.Event) -> None:
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="#e63946", width=2)

    def _on_drag(self, event: tk.Event) -> None:
        if self.rect is not None:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), max(y1, y2)
        self.result = (left, top, right, bottom) if right - left > 20 and bottom - top > 20 else None
        self.overlay.withdraw()
        self.done.set()

    def _on_cancel(self, _event: tk.Event) -> None:
        self.result = None
        self.overlay.withdraw()
        self.done.set()

    def select(self) -> tuple[int, int, int, int] | None:
        self.result = None
        self.done.clear()
        self.overlay.deiconify()
        self.overlay.lift()
        self.overlay.focus_force()
        self.done.wait()
        return self.result


class HotkeyThread(threading.Thread):
    def __init__(self, callback) -> None:
        super().__init__(daemon=True)
        self.callback = callback

    def run(self) -> None:
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(None, HOTKEY_ID, MOD_CONTROL | MOD_ALT, VK_Q):
            return
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:
                self.callback()
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


class ExamAssistantApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Exam Assistant")
        self.root.geometry("980x760")
        self.root.configure(bg="#f6f1e9")

        self.cfg = default_config()
        self.saved = load_saved_settings()
        self.retriever = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.selector = ScreenSelector(root)

        self.api_key_var = tk.StringVar(value=self.saved.get("api_key", ""))
        self.ocr_model_var = tk.StringVar(value=self.saved.get("ocr_model", self.cfg.ocr_model))
        self.answer_model_var = tk.StringVar(value=self.saved.get("answer_model", self.cfg.answer_model))
        self.status_var = tk.StringVar(value="Ready. Press Ctrl+Alt+Q or click Capture.")

        self._build_ui()
        self._load_retriever()
        self._poll_queue()
        HotkeyThread(self._capture_from_hotkey).start()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="OpenRouter key").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.api_key_var, width=70, show="*").grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Save", command=self._save_settings).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="OCR model").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            top,
            textvariable=self.ocr_model_var,
            width=42,
            values=[
                "google/gemini-2.0-flash-001",
                "google/gemini-2.5-flash-image",
                "google/gemini-2.5-pro",
            ],
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(top, text="Answer model").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            top,
            textvariable=self.answer_model_var,
            width=42,
            values=[
                "google/gemini-2.0-flash-001",
                "google/gemini-2.5-flash",
                "google/gemini-2.5-pro",
            ],
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Button(top, text="Capture", command=self._capture_from_button).grid(row=1, column=2, rowspan=2, padx=4)
        top.columnconfigure(1, weight=1)

        status = ttk.Label(self.root, textvariable=self.status_var, padding=(12, 6))
        status.pack(fill="x")

        qwrap = ttk.LabelFrame(self.root, text="Extracted question", padding=12)
        qwrap.pack(fill="both", expand=False, padx=12, pady=(0, 8))
        self.question_box = tk.Text(qwrap, height=7, wrap="word")
        self.question_box.pack(fill="both", expand=True)

        awrap = ttk.LabelFrame(self.root, text="Answer", padding=12)
        awrap.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.answer_box = tk.Text(awrap, wrap="word")
        self.answer_box.pack(fill="both", expand=True)

        swrap = ttk.LabelFrame(self.root, text="Sources", padding=12)
        swrap.pack(fill="both", expand=False, padx=12, pady=(0, 12))
        self.sources_box = tk.Text(swrap, height=10, wrap="word")
        self.sources_box.pack(fill="both", expand=True)

    def _load_retriever(self) -> None:
        self.status_var.set("Loading vector database...")
        self.retriever = Retriever(self.cfg.chroma_dir, self.cfg.collection, self.cfg.embed_model)
        self.status_var.set("Ready. Press Ctrl+Alt+Q or click Capture.")

    def _save_settings(self) -> None:
        save_settings(
            {
                "api_key": self.api_key_var.get().strip(),
                "ocr_model": self.ocr_model_var.get().strip(),
                "answer_model": self.answer_model_var.get().strip(),
            }
        )
        self.status_var.set("Settings saved.")

    def _capture_from_button(self) -> None:
        self.root.after(100, self._capture_flow)

    def _capture_from_hotkey(self) -> None:
        self.root.after(0, self._capture_flow)

    def _capture_flow(self) -> None:
        self.status_var.set("Select the question area on screen.")
        self.root.withdraw()
        bbox = self.selector.select()
        self.root.deiconify()
        self.root.lift()
        if not bbox:
            self.status_var.set("No region selected. Select an area with the exam question.")
            return
        img = ImageGrab.grab(bbox=bbox)
        image_bytes = self._image_to_png_bytes(img)
        self.status_var.set("Running OCR and search...")
        self.answer_box.delete("1.0", "end")
        self.sources_box.delete("1.0", "end")
        worker = threading.Thread(target=self._run_pipeline, args=(image_bytes,), daemon=True)
        worker.start()

    @staticmethod
    def _image_to_png_bytes(img) -> bytes:
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _run_pipeline(self, image_bytes: bytes) -> None:
        api_key = self.api_key_var.get().strip()
        if not api_key:
            self.worker_queue.put(("error", "OpenRouter key is empty."))
            return
        try:
            client = OpenRouterClient(api_key)
            question = client.vision_extract_question(image_bytes, self.ocr_model_var.get().strip())
            if not question or question == "NO_QUESTION_FOUND":
                self.worker_queue.put(("error", "Question was not found. Select the question area more tightly."))
                return
            assert self.retriever is not None
            hits = self.retriever.query(question, self.cfg.top_k)
            answer = client.grounded_answer(question, hits, self.answer_model_var.get().strip())
            self.worker_queue.put(("result", (question, hits, answer)))
        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "error":
                    self.status_var.set("Failed.")
                    messagebox.showerror("Exam Assistant", str(payload))
                else:
                    question, hits, answer = payload
                    self.question_box.delete("1.0", "end")
                    self.question_box.insert("1.0", question)
                    self.answer_box.delete("1.0", "end")
                    self.answer_box.insert("1.0", answer)
                    self.sources_box.delete("1.0", "end")
                    for i, hit in enumerate(hits, start=1):
                        self.sources_box.insert(
                            "end",
                            f"#{i} page={hit.page} dist={hit.distance:.4f}\n{hit.text[:900]}\n\n",
                        )
                    self.status_var.set("Ready.")
        except queue.Empty:
            pass
        self.root.after(250, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ExamAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

