# milcontrol

Python-based access control demo. Train the autoencoder and launch the GUI to
check for anomalous access attempts.

## Setup (Windows)

Run `install.bat` to install TensorFlow and required packages.

## Training

Run the training script to generate the model files and threshold:

```bash
python src/train_autoencoder.py
```

This creates a `model/` folder containing:

* `autoencoder.b64` – base64-encoded autoencoder model
* `encoder.b64` – base64-encoded encoder model
* `threshold.json` – reconstruction error threshold

## Running the GUI

Start the guard program:

```bash
python src/guard_access_gui.py
```

The program loads the saved models and keeps user state in `model/state.json`.
Access logs are stored in `model/log.json`. Use the "최근 기록" and "통계"
buttons in the GUI to view recent entries and summary statistics.
