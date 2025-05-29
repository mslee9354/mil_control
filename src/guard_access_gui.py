import sys
import os

sys.path.append(os.path.dirname(__file__))
import json
import tkinter as tk
from tkinter import messagebox, font

import numpy as np
import tensorflow as tf

from utils import (
    encode_info,
    INPUT_SIZE,
    LATENT_SIZE,
    PURPOSES,
    DESTINATIONS,
    TIMES,
    load_model_b64,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

AUTOENCODER_PATH = os.path.join(MODEL_DIR, 'autoencoder.b64')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.b64')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'threshold.json')
STATE_PATH = os.path.join(MODEL_DIR, 'state.json')
LOG_PATH = os.path.join(MODEL_DIR, 'log.json')


class GuardApp:
    def __init__(self, master):
        self.master = master
        master.title('출입 통제')
        master.geometry('320x260')
        master.resizable(False, False)

        header_font = font.Font(master, size=14, weight='bold')
        label_font = font.Font(master, size=11)

        header = tk.Label(master, text='출입 통제 시스템', font=header_font)
        header.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.id_label = tk.Label(master, text='군번', font=label_font)
        self.id_entry = tk.Entry(master, width=15, font=label_font)

        self.purpose_var = tk.StringVar(master)
        self.purpose_var.set(PURPOSES[0])
        self.purpose_menu = tk.OptionMenu(master, self.purpose_var, *PURPOSES)
        self.purpose_menu.config(width=10, font=label_font)

        self.dest_var = tk.StringVar(master)
        self.dest_var.set(DESTINATIONS[0])
        self.dest_menu = tk.OptionMenu(master, self.dest_var, *DESTINATIONS)
        self.dest_menu.config(width=10, font=label_font)

        self.time_var = tk.StringVar(master)
        self.time_var.set(TIMES[0])
        self.time_menu = tk.OptionMenu(master, self.time_var, *TIMES)
        self.time_menu.config(width=10, font=label_font)

        self.submit_button = tk.Button(master, text='확인', command=self.process, font=label_font, width=14)
        self.recent_button = tk.Button(master, text='최근 기록', command=self.show_recent, font=label_font)
        self.stats_button = tk.Button(master, text='통계', command=self.show_stats, font=label_font)

        self.id_label.grid(row=1, column=0, sticky='e', padx=5)
        self.id_entry.grid(row=1, column=1, sticky='w', pady=2)
        self.purpose_menu.grid(row=2, column=0, columnspan=2, pady=2)
        self.dest_menu.grid(row=3, column=0, columnspan=2, pady=2)
        self.time_menu.grid(row=4, column=0, columnspan=2, pady=2)
        self.submit_button.grid(row=5, column=0, columnspan=2, pady=5)
        self.recent_button.grid(row=6, column=0, pady=2)
        self.stats_button.grid(row=6, column=1, pady=2)

        self.autoencoder = load_model_b64(AUTOENCODER_PATH)
        self.encoder = load_model_b64(ENCODER_PATH)
        with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
            self.threshold = json.load(f)['threshold']

        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                self.state = json.load(f)
        else:
            self.state = {}

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                self.logs = json.load(f)
        else:
            self.logs = []

    def process(self):
        sid = self.id_entry.get().strip()
        if not sid:
            messagebox.showerror('오류', '군번을 입력하세요.')
            return
        if not (sid.startswith('23-76') or sid.startswith('24-76')):
            messagebox.showerror('오류', '군번 형식이 올바르지 않습니다.')
            return

        prev = np.array(self.state.get(sid, [0.0]*LATENT_SIZE))
        info_vec = encode_info(self.purpose_var.get(), self.dest_var.get(), self.time_var.get())
        inp = np.concatenate([prev, info_vec]).reshape(1, -1)
        recon = self.autoencoder.predict(inp, verbose=0)[0]
        error = float(np.mean((recon - inp[0])**2))

        result = '정상' if error <= self.threshold else '이상'
        messagebox.showinfo('결과', f'재구성 오차: {error:.4f}\n판정: {result}')

        new_prev = self.encoder.predict(inp, verbose=0)[0]
        self.state[sid] = new_prev.tolist()
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

        entry = {
            'id': sid,
            'purpose': self.purpose_var.get(),
            'dest': self.dest_var.get(),
            'time': self.time_var.get(),
            'error': error,
            'result': result,
        }
        self.logs.append(entry)
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def show_recent(self):
        sid = self.id_entry.get().strip()
        if not sid:
            messagebox.showerror('오류', '군번을 입력하세요.')
            return
        recent = [log for log in reversed(self.logs) if log['id'] == sid][:5]
        if not recent:
            messagebox.showinfo('최근 기록', '출입 기록이 없습니다.')
            return
        lines = [f"{l['purpose']} {l['dest']} {l['time']} 결과:{l['result']}" for l in recent]
        messagebox.showinfo('최근 기록', '\n'.join(lines))

    def show_stats(self):
        total = len(self.logs)
        normal = sum(1 for l in self.logs if l['result'] == '정상')
        abnormal = total - normal
        messagebox.showinfo('통계', f'총 기록: {total}\n정상: {normal}\n이상: {abnormal}')


if __name__ == '__main__':
    root = tk.Tk()
    app = GuardApp(root)
    root.mainloop()


