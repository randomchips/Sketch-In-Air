# Sketch In Air

Draw through a webcam-tracked marker and recognize hand gestures directly in the browser.

## Technologies

- Python 3.11
- Flask
- OpenCV
- MediaPipe Gesture Recognizer
- NumPy
- HTML, CSS, and JavaScript
- MJPEG video streaming

## Run locally

### Requirements

- 64-bit Python 3.11
- Git
- A working webcam
- Internet access while installing Python packages

### Windows

```powershell
git clone https://github.com/randomchips/Sketch-In-Air.git
cd Sketch-In-Air
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks virtual-environment activation, run this once in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### macOS or Linux

```bash
git clone https://github.com/randomchips/Sketch-In-Air.git
cd Sketch-In-Air
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> after Flask reports that it is running. Stop the server with `Ctrl+C`.

## Use

### Air drawing

1. Open **Air drawing** from the home page.
2. Hold a blue marker or blue object in view of the webcam.
3. Move it through the drawing area to create strokes.
4. Move it over the on-screen controls to change color, undo, clear, pause, or switch between pen and eraser.
5. Use **Save frame** to download the current camera-and-canvas image.

### Hand gestures

1. Open **Hand gestures** from the home page.
2. Show one or two complete hands to the webcam with the fingers visible.
3. The stream displays 21 landmarks per hand, handedness, finger count, gesture name, confidence when available, and a matching emoji.
4. Supported named gestures include closed fist, open palm, pointing up, thumbs up/down, victory, I love you, OK, and call me. Other poses fall back to finger counting.

## Camera and startup issues

- Close other applications using the webcam before opening a studio.
- Allow camera access for Python in the operating-system privacy settings.
- Use front lighting and keep the whole hand inside the frame for best gesture recognition.
- Drawing is calibrated for a blue marker through `penval.npy`.
- The webcam is opened by the computer running Flask; another device on the network only receives the video stream.
- If port `5000` is busy, run:

```bash
python -m flask --app app run --port 5001 --with-threads
```

Then open <http://127.0.0.1:5001>.
