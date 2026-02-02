import os
import sys
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Ensure local imports work in bundled/frozen apps
BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
for path in (BASE_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from config_loader import load_config
    from video_processor import VideoProcessor
except Exception as e:
    raise RuntimeError(f"Project imports failed: {e}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Detector")
        self.geometry("520x420")
        self.resizable(False, False)

        # State
        self.video_path = tk.StringVar()
        self.is_360 = tk.BooleanVar(value=True)
        self.model_size = tk.StringVar(value="m")
        self.frame_skip = tk.IntVar(value=3)
        self.frame_skip_str = tk.StringVar(value="3")
        self.status = tk.StringVar(value="Select a video and press Start")
        self.report_html = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 8}

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Video File").grid(row=0, column=0, sticky="w", **pad)
        ent = ttk.Entry(frm, textvariable=self.video_path, width=44)
        ent.grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Browse", command=self._choose_video).grid(row=0, column=2, **pad)

        ttk.Separator(frm).grid(row=1, column=0, columnspan=3, sticky="we", padx=12)

        ttk.Checkbutton(frm, text="360° Video", variable=self.is_360).grid(row=2, column=0, sticky="w", **pad)

        ttk.Label(frm, text="YOLO Model Size").grid(row=2, column=1, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.model_size, values=["n", "s", "m", "l", "x"], width=6, state="readonly").grid(row=2, column=2, sticky="w", **pad)

        ttk.Label(frm, text="Quality vs Speed").grid(row=3, column=0, sticky="w", **pad)
        ttk.Scale(frm, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.frame_skip, command=self._on_slider_change).grid(row=3, column=1, sticky="we", **pad)
        ttk.Label(frm, textvariable=self.frame_skip_str, width=4, anchor="w").grid(row=3, column=2, sticky="w", **pad)
        ttk.Label(frm, text="1 = Best Quality, 10 = Fastest").grid(row=4, column=1, columnspan=2, sticky="w", padx=12)

        ttk.Separator(frm).grid(row=5, column=0, columnspan=3, sticky="we", padx=12)

        # Actions
        btn_start = ttk.Button(frm, text="Start", command=self._start)
        btn_start.grid(row=6, column=0, **pad)
        btn_report = ttk.Button(frm, text="Open Report", command=self._open_report)
        btn_report.grid(row=6, column=1, **pad)
        btn_quit = ttk.Button(frm, text="Quit", command=self.destroy)
        btn_quit.grid(row=6, column=2, **pad)

        # Status + progress
        ttk.Label(frm, textvariable=self.status, foreground="#333").grid(row=7, column=0, columnspan=3, sticky="w", padx=12)
        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.grid(row=8, column=0, columnspan=3, sticky="we", padx=12)
        self.progress_pct = tk.StringVar(value="0%")
        ttk.Label(frm, textvariable=self.progress_pct).grid(row=9, column=2, sticky="e", padx=12)

        # Make columns flexible
        for c in range(3):
            frm.grid_columnconfigure(c, weight=1)

    def _choose_video(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=[
            ("Video Files", ".mp4 .avi .mov .mkv .flv .wmv"),
            ("All Files", "*.*"),
        ])
        if path:
            self.video_path.set(path)

    def _open_report(self):
        if self.report_html and os.path.exists(self.report_html):
            webbrowser.open(f"file://{os.path.abspath(self.report_html)}")
        else:
            messagebox.showinfo("Report", "No report yet. Run processing first.")

    def _start(self):
        video = self.video_path.get().strip().strip('"').strip("'")
        if not video or not os.path.exists(video):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        self.status.set("Processing...")
        self.progress.configure(value=0)
        self.progress_pct.set("0%")

        t = threading.Thread(target=self._process_video, args=(video,), daemon=True)
        t.start()

    def _process_video(self, video_path: str):
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            
            # Pre-flight check: verify output directory is writable before processing
            try:
                from config_loader import load_config
                cfg_loader = load_config(base_dir)
                output_cfg = cfg_loader.get_output_config()
                # Force output to data/output folder (absolute path)
                output_dir = os.path.join(base_dir, 'data', 'output')
                os.makedirs(output_dir, exist_ok=True)
                if not os.access(output_dir, os.W_OK):
                    raise PermissionError(f"No write permission for output directory: {output_dir}")
            except Exception as e:
                self.after(0, lambda msg=str(e): messagebox.showerror("Setup Error", f"Cannot set up output: {msg}"))
                self.status.set("Setup error.")
                return
            
            # Now proceed with processing
            cfg_loader = load_config(base_dir)
            cfg = cfg_loader.get_processing_config()

            # Apply UI selections
            cfg['is_360'] = bool(self.is_360.get())
            cfg['yolo_model'] = self.model_size.get().strip() or cfg['yolo_model']
            # Clamp frame_skip between 1 and 10
            fs = int(self.frame_skip.get())
            cfg['frame_skip'] = max(1, min(10, fs))

            output_cfg = cfg_loader.get_output_config()
            # Force output to data/output folder (absolute path)
            output_dir = os.path.join(base_dir, 'data', 'output')
            os.makedirs(output_dir, exist_ok=True)

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}_processed.avi")

            # Progress callback to update UI safely from background thread
            def on_progress(data: dict):
                try:
                    percent = int(data.get('percent', 0))
                except Exception:
                    percent = 0
                self.after(0, lambda: self._update_progress(percent))

            processor = VideoProcessor(
                is_360_video=cfg['is_360'],
                frame_skip=cfg['frame_skip'],
                debug_mode=cfg['debug_mode'],
                process_percentage=cfg['process_percentage'],
                min_hits=cfg['min_hits'],
                model_size=cfg['yolo_model'],
                min_object_size=cfg['min_object_size'],
                max_distance_ratio=cfg['max_distance_ratio'],
                show_progress=False,
                progress_callback=on_progress,
                progress_every=20,
            )
            processor.confidence_threshold = cfg['confidence_threshold']
            if hasattr(processor, 'detector') and processor.detector is not None:
                processor.detector.confidence_threshold = cfg['confidence_threshold']

            results = processor.process_video(video_path, output_path)
            if not results:
                raise RuntimeError("No results returned.")

            # Save Power BI-ready CSVs (match CLI output behavior)
            try:
                from main import save_results_to_csv, save_detailed_counts_to_csv

                if output_cfg.get('save_csv', True):
                    csv_path = os.path.splitext(output_path)[0] + "_results.csv"
                    save_results_to_csv(results, csv_path)

                if output_cfg.get('save_detailed_counts', True):
                    detailed_counts = processor.get_detailed_counts()
                    detailed_csv_path = os.path.splitext(output_path)[0] + "_detailed_counts.csv"
                    save_detailed_counts_to_csv(detailed_counts, detailed_csv_path)
            except Exception as e:
                self.after(0, lambda msg=str(e): messagebox.showwarning("CSV Warning", f"Failed to save CSV: {msg}"))

            # Generate reports via main's helper (re-import to avoid circular issues)
            from main import generate_summary_report
            report_paths = generate_summary_report(results, video_path, cfg, output_dir)

            # Determine HTML path
            html_path = None
            if isinstance(report_paths, tuple):
                _, html_path = report_paths
            elif isinstance(report_paths, str) and report_paths.endswith('.html'):
                html_path = report_paths

            self.report_html = html_path
            self.status.set("Done. Reports saved in data/output.")
            self.after(0, lambda: self._update_progress(100))

            # Auto-open HTML report if available
            if html_path and os.path.exists(html_path):
                webbrowser.open(f"file://{os.path.abspath(html_path)}")

        except Exception as e:
            self.after(0, lambda: self._update_progress(0))
            self.status.set("Error. See details.")
            messagebox.showerror("Processing Error", str(e))

    def _update_progress(self, percent: int):
        percent = max(0, min(100, int(percent)))
        self.progress.configure(value=percent)
        self.progress_pct.set(f"{percent}%")

    def _on_slider_change(self, val):
        try:
            v = int(float(val))
        except Exception:
            v = self.frame_skip.get()
        self.frame_skip_str.set(str(v))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()