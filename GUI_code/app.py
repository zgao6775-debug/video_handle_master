import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from pipelines import (
    ensure_dir,
    repo_root,
    task_smart_bgm,
    task_auto_effects,
    task_burn_subtitles_from_unified,
    task_motion_analysis,
    task_object_detection_yolo,
    task_unified_report,
    timestamp_id,
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("视频智能处理系统")
        self.geometry("1080x880")

        self.root_dir = repo_root()
        self.default_output_dir = os.path.join(self.root_dir, "output")
        ensure_dir(self.default_output_dir)

        self.video_path = tk.StringVar()
        self.out_base = tk.StringVar(value=self.default_output_dir)
        self.run_id = tk.StringVar(value=timestamp_id())

        self.unified_enabled = tk.BooleanVar(value=True)
        self.language_preset = tk.StringVar(value="生成英文字幕/摘要（推荐）")
        self.quality_preset = tk.StringVar(value="标准（推荐）")
        self.with_srt = tk.BooleanVar(value=True)
        self.use_dashscope = tk.BooleanVar(value=False)

        self.effects_enabled = tk.BooleanVar(value=True)
        self.effects_fade = tk.DoubleVar(value=0.4)

        self.motion_enabled = tk.BooleanVar(value=False)
        self.motion_max_frames = tk.StringVar(value="")
        self.motion_skip_model = tk.BooleanVar(value=False)

        self.yolo_enabled = tk.BooleanVar(value=False)

        self.soundtrack_enabled = tk.BooleanVar(value=False)
        default_lib = os.path.join(self.root_dir, "video_BGM")
        if not os.path.isdir(default_lib):
            default_lib = os.path.join(self.root_dir, "music")
        if not os.path.isdir(default_lib):
            default_lib = os.path.join(self.root_dir, "Video_BGM")
        self.bgm_library_dir = tk.StringVar(value=default_lib)
        self.bgm_volume = tk.DoubleVar(value=0.7)

        self.burn_enabled = tk.BooleanVar(value=False)
        self.burn_lang = tk.StringVar(value="原语言字幕（自动）")

        self._build_ui()
        self._worker = None

    def _targets_value(self):
        preset = self.language_preset.get().strip()
        mapping = {
            "不翻译（仅中文）": "",
            "生成英文字幕/摘要（推荐）": "en",
            "生成英文+日文摘要（需联网增强）": "en,ja",
            "生成英文+法文摘要（需联网增强）": "en,fr",
        }
        return mapping.get(preset, "en")

    def _whisper_model_value(self):
        preset = self.quality_preset.get().strip()
        mapping = {
            "标准（推荐）": "small",
            "更精确（更慢）": "medium",
        }
        return mapping.get(preset, "small")

    def _build_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")

        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        ttk.Label(row1, text="输入视频:").pack(side="left")
        ttk.Entry(row1, textvariable=self.video_path, width=78).pack(side="left", padx=8)
        ttk.Button(row1, text="选择...", command=self._choose_video).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="输出目录:").pack(side="left")
        ttk.Entry(row2, textvariable=self.out_base, width=78).pack(side="left", padx=8)
        ttk.Button(row2, text="选择...", command=self._choose_out).pack(side="left")

        row3 = ttk.Frame(top)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Label(row3, text="Run ID:").pack(side="left")
        ttk.Entry(row3, textvariable=self.run_id, width=30).pack(side="left", padx=8)
        ttk.Button(row3, text="刷新", command=lambda: self.run_id.set(timestamp_id())).pack(side="left")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=8)

        tab_main = ttk.Frame(nb, padding=12)
        tab_unified = ttk.Frame(nb, padding=12)
        tab_effects = ttk.Frame(nb, padding=12)
        tab_motion = ttk.Frame(nb, padding=12)
        tab_yolo = ttk.Frame(nb, padding=12)
        tab_music = ttk.Frame(nb, padding=12)
        tab_burn = ttk.Frame(nb, padding=12)

        nb.add(tab_main, text="一键运行")
        nb.add(tab_unified, text="关键帧+摘要")
        nb.add(tab_effects, text="自动特效")
        nb.add(tab_motion, text="动作分析")
        nb.add(tab_yolo, text="目标检测")
        nb.add(tab_music, text="自动配乐")
        nb.add(tab_burn, text="字幕烧录")

        self._build_tab_main(tab_main)
        self._build_tab_unified(tab_unified)
        self._build_tab_effects(tab_effects)
        self._build_tab_motion(tab_motion)
        self._build_tab_yolo(tab_yolo)
        self._build_tab_music(tab_music)
        self._build_tab_burn(tab_burn)

        bottom = ttk.Frame(self, padding=12)
        bottom.pack(fill="both", expand=True)

        btn_row = ttk.Frame(bottom)
        btn_row.pack(fill="x")
        self.run_btn = ttk.Button(btn_row, text="开始运行", command=self._start_run)
        self.run_btn.pack(side="left")
        ttk.Button(btn_row, text="打开输出目录", command=self._open_output_dir).pack(side="left", padx=8)

        self.log_box = tk.Text(bottom, height=18, wrap="word")
        self.log_box.pack(fill="both", expand=True, pady=(10, 0))

    def _build_tab_main(self, parent):
        ttk.Label(parent, text="选择要执行的功能（会写入 output/RunID/ 下）").pack(anchor="w")

        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=10)

        ttk.Checkbutton(grid, text="关键帧+摘要（生成报告+关键帧）", variable=self.unified_enabled).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(grid, text="自动特效", variable=self.effects_enabled).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Checkbutton(grid, text="动作分析", variable=self.motion_enabled).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Checkbutton(grid, text="目标检测（识别画面里的物体）", variable=self.yolo_enabled).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Checkbutton(grid, text="自动配乐（给视频配上背景音乐）", variable=self.soundtrack_enabled).grid(row=4, column=0, sticky="w", pady=4)
        ttk.Checkbutton(grid, text="字幕压到视频里（需要先生成字幕）", variable=self.burn_enabled).grid(row=5, column=0, sticky="w", pady=4)

        opts = ttk.LabelFrame(parent, text="关键帧+摘要设置", padding=10)
        opts.pack(fill="x", pady=8)
        r1 = ttk.Frame(opts)
        r1.pack(fill="x")
        ttk.Label(r1, text="翻译/多语言:").pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self.language_preset,
            values=[
                "不翻译（仅中文）",
                "生成英文字幕/摘要（推荐）",
                "生成英文+日文摘要（需联网增强）",
                "生成英文+法文摘要（需联网增强）",
            ],
            width=28,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Label(r1, text="识别精度:").pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self.quality_preset,
            values=["标准（推荐）", "更精确（更慢）"],
            width=14,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Checkbutton(r1, text="生成字幕文件（SRT）", variable=self.with_srt).pack(side="left", padx=8)
        ttk.Checkbutton(r1, text="联网增强（摘要更好/更多语言）", variable=self.use_dashscope).pack(side="left", padx=8)

        tip = ttk.Label(
            opts,
            text="提示：联网增强需要可用的 API Key；未开启也能生成关键帧与基础摘要。",
        )
        tip.pack(anchor="w", pady=(8, 0))

        music_opts = ttk.LabelFrame(parent, text="自动配乐设置", padding=10)
        music_opts.pack(fill="x", pady=8)
        r1 = ttk.Frame(music_opts)
        r1.pack(fill="x", pady=4)
        ttk.Label(r1, text="方式:").pack(side="left")
        ttk.Label(r1, text="智能匹配（本地曲库，无需联网）").pack(side="left", padx=8)

        r2 = ttk.Frame(music_opts)
        r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="智能配乐曲库目录:").pack(side="left")
        ttk.Entry(r2, textvariable=self.bgm_library_dir, width=70).pack(side="left", padx=8)
        ttk.Button(r2, text="选择...", command=self._choose_bgm_library).pack(side="left")

        r3 = ttk.Frame(music_opts)
        r3.pack(fill="x", pady=4)
        ttk.Label(r3, text="智能配乐音量系数:").pack(side="left")
        ttk.Entry(r3, textvariable=self.bgm_volume, width=10).pack(side="left", padx=8)

    def _build_tab_unified(self, parent):
        r1 = ttk.Frame(parent)
        r1.pack(fill="x")
        ttk.Label(r1, text="翻译/多语言:").pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self.language_preset,
            values=[
                "不翻译（仅中文）",
                "生成英文字幕/摘要（推荐）",
                "生成英文+日文摘要（需联网增强）",
                "生成英文+法文摘要（需联网增强）",
            ],
            width=30,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Label(r1, text="识别精度:").pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self.quality_preset,
            values=["标准（推荐）", "更精确（更慢）"],
            width=14,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Checkbutton(r1, text="生成字幕文件（SRT）", variable=self.with_srt).pack(side="left", padx=8)
        ttk.Checkbutton(r1, text="联网增强（摘要更好/更多语言）", variable=self.use_dashscope).pack(side="left", padx=8)
        ttk.Button(parent, text="仅运行该功能", command=self._start_run_unified).pack(anchor="w", pady=(12, 0))

    def _build_tab_effects(self, parent):
        ttk.Checkbutton(parent, text="启用自动特效", variable=self.effects_enabled).pack(anchor="w")
        r1 = ttk.Frame(parent)
        r1.pack(fill="x", pady=8)
        ttk.Label(r1, text="转场淡入淡出(秒):").pack(side="left")
        ttk.Entry(r1, textvariable=self.effects_fade, width=10).pack(side="left", padx=8)
        ttk.Button(parent, text="仅运行该功能", command=self._start_run_effects).pack(anchor="w", pady=(12, 0))

    def _build_tab_motion(self, parent):
        ttk.Checkbutton(parent, text="启用动作分析", variable=self.motion_enabled).pack(anchor="w")
        r1 = ttk.Frame(parent)
        r1.pack(fill="x", pady=8)
        ttk.Label(r1, text="最大帧数(可空):").pack(side="left")
        ttk.Entry(r1, textvariable=self.motion_max_frames, width=12).pack(side="left", padx=8)
        ttk.Checkbutton(r1, text="跳过模型(无检测)", variable=self.motion_skip_model).pack(side="left", padx=8)
        ttk.Button(parent, text="仅运行该功能", command=self._start_run_motion).pack(anchor="w", pady=(12, 0))

    def _build_tab_yolo(self, parent):
        ttk.Checkbutton(parent, text="启用YOLOv3目标检测", variable=self.yolo_enabled).pack(anchor="w")
        ttk.Label(parent, text="该功能会生成带框视频(avi)，耗时较长。").pack(anchor="w", pady=(8, 0))
        ttk.Button(parent, text="仅运行该功能", command=self._start_run_yolo).pack(anchor="w", pady=(12, 0))

    def _build_tab_music(self, parent):
        ttk.Checkbutton(parent, text="启用自动配乐", variable=self.soundtrack_enabled).pack(anchor="w")
        r1 = ttk.Frame(parent)
        r1.pack(fill="x", pady=8)
        ttk.Label(r1, text="方式:").pack(side="left")
        ttk.Label(r1, text="智能匹配（本地曲库，无需联网）").pack(side="left", padx=8)

        r2 = ttk.Frame(parent)
        r2.pack(fill="x", pady=8)
        ttk.Label(r2, text="智能配乐曲库目录:").pack(side="left")
        ttk.Entry(r2, textvariable=self.bgm_library_dir, width=70).pack(side="left", padx=8)
        ttk.Button(r2, text="选择...", command=self._choose_bgm_library).pack(side="left")

        r3 = ttk.Frame(parent)
        r3.pack(fill="x", pady=8)
        ttk.Label(r3, text="智能配乐音量系数:").pack(side="left")
        ttk.Entry(r3, textvariable=self.bgm_volume, width=10).pack(side="left", padx=8)

        ttk.Label(parent, text="输出：video_with_music.mp4 + meta(json)").pack(anchor="w", pady=(8, 0))
        ttk.Button(parent, text="仅运行该功能", command=self._start_run_music).pack(anchor="w", pady=(12, 0))

    def _build_tab_burn(self, parent):
        ttk.Checkbutton(parent, text="启用字幕压到视频里（需要先生成字幕）", variable=self.burn_enabled).pack(anchor="w")
        r1 = ttk.Frame(parent)
        r1.pack(fill="x", pady=8)
        ttk.Label(r1, text="选择字幕语言:").pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self.burn_lang,
            values=["原语言字幕（自动）", "英文字幕"],
            width=16,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Button(parent, text="仅运行该功能", command=self._start_run_burn).pack(anchor="w", pady=(12, 0))

    def _burn_lang_value(self):
        v = self.burn_lang.get().strip()
        if v == "英文字幕":
            return "en"
        return "auto"

    def _dummy_true(self):
        v = tk.BooleanVar(value=True)
        return v

    def _choose_video(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")])
        if p:
            self.video_path.set(p)

    def _choose_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_base.set(p)

    def _choose_bgm_library(self):
        p = filedialog.askdirectory()
        if p:
            self.bgm_library_dir.set(p)

    def _open_output_dir(self):
        path = os.path.join(self.out_base.get(), self.run_id.get())
        ensure_dir(path)
        try:
            os.startfile(path)
        except Exception:
            messagebox.showinfo("提示", f"输出目录：{path}")

    def log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def _start_run(self):
        self._run_pipeline(mode="all")

    def _start_run_unified(self):
        self._run_pipeline(mode="unified")

    def _start_run_effects(self):
        self._run_pipeline(mode="effects")

    def _start_run_motion(self):
        self._run_pipeline(mode="motion")

    def _start_run_yolo(self):
        self._run_pipeline(mode="yolo")

    def _start_run_burn(self):
        self._run_pipeline(mode="burn")

    def _start_run_music(self):
        self._run_pipeline(mode="music")

    def _ensure_soundtrack_inputs(self, mode):
        will_run = (mode == "music") or (mode == "all" and self.soundtrack_enabled.get())
        if not will_run:
            return True

        lib_dir = self.bgm_library_dir.get().strip()
        if not lib_dir:
            self._choose_bgm_library()
            lib_dir = self.bgm_library_dir.get().strip()
        if not lib_dir:
            messagebox.showerror("错误", "请选择智能配乐曲库目录（包含音频文件）")
            return False
        if not os.path.isdir(lib_dir):
            messagebox.showerror("错误", "曲库目录不存在，请重新选择")
            return False
        try:
            v = float(self.bgm_volume.get())
            if v <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("错误", "音量系数必须是大于 0 的数字，例如 0.7")
            return False
        return True

    def _validate(self):
        vp = self.video_path.get().strip()
        if not vp or not os.path.exists(vp):
            messagebox.showerror("错误", "请先选择有效的视频文件")
            return None
        base = self.out_base.get().strip()
        if not base:
            messagebox.showerror("错误", "请设置输出目录")
            return None
        rid = self.run_id.get().strip() or timestamp_id()
        self.run_id.set(rid)
        return vp

    def _run_pipeline(self, mode):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("提示", "任务正在运行中")
            return
        vp = self._validate()
        if not vp:
            return
        if not self._ensure_soundtrack_inputs(mode):
            return
        self.run_btn.configure(state="disabled")
        self.log("开始运行...")
        self.log(f"视频: {vp}")

        def work():
            try:
                run_dir = os.path.join(self.out_base.get(), self.run_id.get())
                ensure_dir(run_dir)

                unified_dir = os.path.join(run_dir, "unified_output")
                effects_dir = os.path.join(run_dir, "effects")
                motion_dir = os.path.join(run_dir, "motion")
                yolo_dir = os.path.join(run_dir, "yolo")
                music_dir = os.path.join(run_dir, "soundtrack")
                burn_dir = os.path.join(run_dir, "subtitled")

                if mode in ("all", "unified", "burn"):
                    need_unified = (mode in ("unified", "burn")) or bool(self.unified_enabled.get())
                    if need_unified:
                        task_unified_report(
                            vp,
                            unified_dir,
                            targets=self._targets_value(),
                            whisper_model=self._whisper_model_value(),
                            with_srt=(True if (mode == "burn" or self.burn_enabled.get()) else bool(self.with_srt.get())),
                            use_dashscope=bool(self.use_dashscope.get()),
                            log=self.log,
                        )

                if mode == "effects" or (mode == "all" and self.effects_enabled.get()):
                    task_auto_effects(
                        vp,
                        os.path.join(effects_dir, "output_with_effects.mp4"),
                        os.path.join(effects_dir, "effects_report.json"),
                        transition_fade=float(self.effects_fade.get()),
                        log=self.log,
                    )

                if mode == "motion" or (mode == "all" and self.motion_enabled.get()):
                    mf = self.motion_max_frames.get().strip()
                    max_frames = int(mf) if mf else None
                    task_motion_analysis(
                        vp,
                        os.path.join(motion_dir, "motion_report.json"),
                        max_frames=max_frames,
                        skip_model=bool(self.motion_skip_model.get()),
                        no_display=True,
                        log=self.log,
                    )

                if mode == "yolo" or (mode == "all" and self.yolo_enabled.get()):
                    task_object_detection_yolo(vp, yolo_dir, log=self.log)

                if mode == "music" or (mode == "all" and self.soundtrack_enabled.get()):
                    self.log("开始自动配乐（生成背景音乐并合成视频）...")
                    task_smart_bgm(
                        vp,
                        music_dir,
                        library_dir=self.bgm_library_dir.get(),
                        volume=float(self.bgm_volume.get()),
                        log=self.log,
                        unified_dir=unified_dir,
                    )

                if mode == "burn" or (mode == "all" and self.burn_enabled.get()):
                    self.log("开始字幕压入（生成带字幕的新视频）...")
                    task_burn_subtitles_from_unified(
                        unified_dir,
                        vp,
                        self._burn_lang_value(),
                        os.path.join(burn_dir, f"subtitled_{self._burn_lang_value()}.mp4"),
                        log=self.log,
                    )

                self.log("完成。")
                self.log(f"输出目录: {run_dir}")
            except Exception as e:
                self.log(f"失败: {e}")
                messagebox.showerror("失败", str(e))
            finally:
                self.run_btn.configure(state="normal")

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

