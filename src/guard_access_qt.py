from __future__ import annotations

import os
import json
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

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


class GuardApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('출입 통제')
        self.setFixedSize(320, 260)
        self._build_ui()

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

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel('출입 통제 시스템')
        header_font = QtGui.QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(header)

        form = QtWidgets.QGridLayout()
        layout.addLayout(form)

        label_font = QtGui.QFont()
        label_font.setPointSize(11)

        id_label = QtWidgets.QLabel('군번')
        id_label.setFont(label_font)
        self.id_entry = QtWidgets.QLineEdit()
        self.id_entry.setFont(label_font)
        form.addWidget(id_label, 0, 0)
        form.addWidget(self.id_entry, 0, 1)

        self.purpose_box = QtWidgets.QComboBox()
        self.purpose_box.addItems(PURPOSES)
        self.purpose_box.setFont(label_font)
        form.addWidget(self.purpose_box, 1, 0, 1, 2)

        self.dest_box = QtWidgets.QComboBox()
        self.dest_box.addItems(DESTINATIONS)
        self.dest_box.setFont(label_font)
        form.addWidget(self.dest_box, 2, 0, 1, 2)

        self.time_box = QtWidgets.QComboBox()
        self.time_box.addItems(TIMES)
        self.time_box.setFont(label_font)
        form.addWidget(self.time_box, 3, 0, 1, 2)

        self.submit_btn = QtWidgets.QPushButton('확인')
        self.submit_btn.setFont(label_font)
        self.submit_btn.clicked.connect(self.process)
        layout.addWidget(self.submit_btn)

        btn_row = QtWidgets.QHBoxLayout()
        self.recent_btn = QtWidgets.QPushButton('최근 기록')
        self.recent_btn.setFont(label_font)
        self.recent_btn.clicked.connect(self.show_recent)
        btn_row.addWidget(self.recent_btn)
        self.stats_btn = QtWidgets.QPushButton('통계')
        self.stats_btn.setFont(label_font)
        self.stats_btn.clicked.connect(self.show_stats)
        btn_row.addWidget(self.stats_btn)
        layout.addLayout(btn_row)

    # core logic - same as Tkinter version
    def process(self):
        sid = self.id_entry.text().strip()
        if not sid:
            QtWidgets.QMessageBox.critical(self, '오류', '군번을 입력하세요.')
            return
        if not (sid.startswith('23-76') or sid.startswith('24-76')):
            QtWidgets.QMessageBox.critical(self, '오류', '군번 형식이 올바르지 않습니다.')
            return

        prev = np.array(self.state.get(sid, [0.0] * LATENT_SIZE))
        info_vec = encode_info(
            self.purpose_box.currentText(),
            self.dest_box.currentText(),
            self.time_box.currentText(),
        )
        inp = np.concatenate([prev, info_vec]).reshape(1, -1)
        recon = self.autoencoder.predict(inp, verbose=0)[0]
        error = float(np.mean((recon - inp[0]) ** 2))
        result = '정상' if error <= self.threshold else '이상'
        QtWidgets.QMessageBox.information(
            self, '결과', f'재구성 오차: {error:.4f}\n판정: {result}'
        )

        new_prev = self.encoder.predict(inp, verbose=0)[0]
        self.state[sid] = new_prev.tolist()
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

        entry = {
            'id': sid,
            'purpose': self.purpose_box.currentText(),
            'dest': self.dest_box.currentText(),
            'time': self.time_box.currentText(),
            'error': error,
            'result': result,
        }
        self.logs.append(entry)
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def show_recent(self):
        sid = self.id_entry.text().strip()
        if not sid:
            QtWidgets.QMessageBox.critical(self, '오류', '군번을 입력하세요.')
            return
        recent = [log for log in reversed(self.logs) if log['id'] == sid][:5]
        if not recent:
            QtWidgets.QMessageBox.information(self, '최근 기록', '출입 기록이 없습니다.')
            return
        lines = [
            f"{l['purpose']} {l['dest']} {l['time']} 결과:{l['result']}" for l in recent
        ]
        QtWidgets.QMessageBox.information(self, '최근 기록', '\n'.join(lines))

    def show_stats(self):
        total = len(self.logs)
        normal = sum(1 for l in self.logs if l['result'] == '정상')
        abnormal = total - normal
        QtWidgets.QMessageBox.information(
            self,
            '통계',
            f'총 기록: {total}\n정상: {normal}\n이상: {abnormal}',
        )


def main():
    app = QtWidgets.QApplication([])
    QtWidgets.QApplication.setStyle('Fusion')

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197).lighter())
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)
    window = GuardApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
