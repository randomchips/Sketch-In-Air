import atexit
import cv2
from gesture_recognition import GestureEngine
import numpy as np
import os
from collections import deque
from flask import Flask,render_template,Response

app=Flask(__name__)

def contour_center(contour):
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return None
    return (
        int(moments['m10'] / moments['m00']),
        int(moments['m01'] / moments['m00'])
    )

def smooth_marker_point(points):
    point_array = np.asarray(points, dtype=np.float32)
    weights = np.arange(1, len(point_array) + 1, dtype=np.float32)
    return tuple(np.average(point_array, axis=0, weights=weights).astype(int))

def drawing_control_layout(frame_width, frame_height):
    margin = max(6, frame_width // 100)
    gap = max(4, frame_width // 160)
    top_height = max(48, int(frame_height * 0.11))
    actions = ('clear', 'undo', 'color_0', 'color_1', 'color_2', 'color_3')
    usable_width = frame_width - (2 * margin) - (gap * (len(actions) - 1))
    button_width = usable_width // len(actions)
    controls = {}

    for index, action in enumerate(actions):
        left = margin + index * (button_width + gap)
        right = frame_width - margin if index == len(actions) - 1 else left + button_width
        controls[action] = (left, margin, right, margin + top_height)

    bottom_height = max(46, int(frame_height * 0.1))
    bottom_width = max(92, int(frame_width * 0.16))
    bottom_top = frame_height - margin - bottom_height
    controls['mode'] = (margin, bottom_top, margin + bottom_width, frame_height - margin)
    controls['pause'] = (
        frame_width - margin - bottom_width,
        bottom_top,
        frame_width - margin,
        frame_height - margin
    )
    return controls

def control_at_point(point, controls):
    x_coordinate, y_coordinate = point
    for action, (left, top, right, bottom) in controls.items():
        if left <= x_coordinate <= right and top <= y_coordinate <= bottom:
            return action
    return None

def draw_control_button(frame, rectangle, label, fill_color, text_color, active=False):
    left, top, right, bottom = rectangle
    border_color = (255, 255, 255) if active else (30, 35, 42)
    border_width = 3 if active else 1
    cv2.rectangle(frame, (left, top), (right, bottom), fill_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), border_color, border_width, cv2.LINE_AA)

    font_scale = max(0.32, min(0.52, (right - left) / 180))
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    text_x = left + max(4, ((right - left) - text_size[0]) // 2)
    text_y = top + ((bottom - top) + text_size[1]) // 2
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA
    )

def draw_drawing_controls(frame, controls, colors, color_index, mode, paused):
    button_styles = {
        'clear': ('CLEAR', (68, 73, 80), (255, 255, 255)),
        'undo': ('UNDO', (68, 73, 80), (255, 255, 255)),
        'color_0': ('BLUE', colors[0], (255, 255, 255)),
        'color_1': ('GREEN', colors[1], (20, 35, 24)),
        'color_2': ('RED', colors[2], (255, 255, 255)),
        'color_3': ('YELLOW', colors[3], (30, 35, 20))
    }
    for action, (label, fill_color, text_color) in button_styles.items():
        draw_control_button(
            frame,
            controls[action],
            label,
            fill_color,
            text_color,
            action == f'color_{color_index}'
        )

    draw_control_button(
        frame,
        controls['mode'],
        'ERASER' if mode == 'eraser' else 'PEN',
        (82, 88, 96),
        (255, 255, 255),
        mode == 'eraser'
    )
    draw_control_button(
        frame,
        controls['pause'],
        'RESUME' if paused else 'PAUSE',
        (57, 95, 105) if paused else (82, 88, 96),
        (255, 255, 255),
        paused
    )

def encode_mjpeg_frame(frame):
    encoded, buffer = cv2.imencode(
        '.jpg',
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 88]
    )
    if not encoded:
        return None
    return (
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    )

def camera_error_frame(message):
    frame = np.full((480, 640, 3), (28, 32, 38), dtype=np.uint8)
    text_size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = max(20, (frame.shape[1] - text_size[0]) // 2)
    cv2.putText(
        frame,
        message,
        (text_x, frame.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA
    )
    return frame

def updated_generate_frames():
    pen_values = np.load(os.path.join(os.path.dirname(__file__), 'penval.npy'))
    lower_range = pen_values[0].astype(np.uint8)
    upper_range = pen_values[1].astype(np.uint8)
    colors = [(255, 80, 35), (55, 220, 105), (45, 65, 240), (40, 220, 245)]
    kernel = np.ones((5, 5), np.uint8)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    canvas = None
    previous_point = None
    smoothed_points = deque(maxlen=5)
    history = []
    active_control = None
    stroke_started = False
    color_index = 0
    mode = 'pen'
    paused = False

    try:
        if not capture.isOpened():
            payload = encode_mjpeg_frame(camera_error_frame('Camera is unavailable'))
            if payload is not None:
                yield payload
            return

        while True:
            frame_read, frame = capture.read()
            if not frame_read or frame is None:
                payload = encode_mjpeg_frame(camera_error_frame('Camera frame was lost'))
                if payload is not None:
                    yield payload
                return

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            if canvas is None or canvas.shape != frame.shape:
                canvas = np.zeros_like(frame)
                history.clear()

            controls = drawing_control_layout(frame_width, frame_height)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            marker_mask = cv2.inRange(hsv_frame, lower_range, upper_range)
            marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)
            marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(
                marker_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cursor_point = None
            cursor_radius = 10
            valid_marker = False
            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(contour)
                minimum_area = max(500, frame_width * frame_height * 0.0015)
                maximum_area = frame_width * frame_height * 0.35
                raw_point = contour_center(contour)
                valid_marker = minimum_area <= contour_area <= maximum_area and raw_point is not None

                if valid_marker:
                    cursor_point = raw_point
                    cursor_radius = max(9, min(24, int(np.sqrt(contour_area) / 3)))
                    hovered_control = control_at_point(raw_point, controls)

                    if hovered_control is not None:
                        if hovered_control != active_control:
                            if hovered_control == 'clear' and np.any(canvas):
                                history.append(canvas.copy())
                                history = history[-20:]
                                canvas.fill(0)
                            elif hovered_control == 'undo' and history:
                                canvas = history.pop()
                            elif hovered_control.startswith('color_'):
                                color_index = int(hovered_control[-1])
                                mode = 'pen'
                            elif hovered_control == 'mode':
                                mode = 'eraser' if mode == 'pen' else 'pen'
                            elif hovered_control == 'pause':
                                paused = not paused

                        active_control = hovered_control
                        previous_point = None
                        stroke_started = False
                        smoothed_points.clear()
                    else:
                        active_control = None
                        smoothed_points.append(raw_point)
                        cursor_point = smooth_marker_point(smoothed_points)

                        if paused:
                            previous_point = None
                            stroke_started = False
                        elif previous_point is None:
                            previous_point = cursor_point
                        else:
                            distance = float(np.hypot(
                                cursor_point[0] - previous_point[0],
                                cursor_point[1] - previous_point[1]
                            ))
                            maximum_jump = max(45, min(frame_width, frame_height) * 0.18)

                            if distance > maximum_jump:
                                stroke_started = False
                            elif distance >= 2:
                                if not stroke_started:
                                    history.append(canvas.copy())
                                    history = history[-20:]
                                    stroke_started = True

                                if mode == 'pen':
                                    line_width = max(4, int(min(frame_width, frame_height) * 0.01))
                                    cv2.line(
                                        canvas,
                                        previous_point,
                                        cursor_point,
                                        colors[color_index],
                                        line_width,
                                        cv2.LINE_AA
                                    )
                                else:
                                    eraser_width = max(24, int(min(frame_width, frame_height) * 0.06))
                                    cv2.line(
                                        canvas,
                                        previous_point,
                                        cursor_point,
                                        (0, 0, 0),
                                        eraser_width,
                                        cv2.LINE_AA
                                    )

                            previous_point = cursor_point

            if not valid_marker:
                active_control = None
                previous_point = None
                stroke_started = False
                smoothed_points.clear()

            painted_pixels = np.any(canvas != 0, axis=2)
            frame[painted_pixels] = canvas[painted_pixels]
            draw_drawing_controls(frame, controls, colors, color_index, mode, paused)

            if cursor_point is not None:
                cursor_color = (255, 255, 255) if mode == 'eraser' else colors[color_index]
                cv2.circle(frame, cursor_point, cursor_radius, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.circle(frame, cursor_point, 3, cursor_color, -1, cv2.LINE_AA)

            payload = encode_mjpeg_frame(frame)
            if payload is not None:
                yield payload
    finally:
        capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        updated_generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



gesture_engine = GestureEngine(
    model_path=os.path.join(
        os.path.dirname(__file__),
        'emojinator_files',
        'gesture_recognizer.task',
    ),
    emoji_directory=os.path.join(
        os.path.dirname(__file__),
        'emojinator_files',
        'hand_emo',
    ),
)
atexit.register(gesture_engine.close)


def generate_emoji(capture, engine):
    try:
        if not capture.isOpened():
            payload = encode_mjpeg_frame(camera_error_frame('Camera is unavailable'))
            if payload is not None:
                yield payload
            return

        while True:
            frame_read, frame = capture.read()
            if not frame_read or frame is None:
                payload = encode_mjpeg_frame(camera_error_frame('Camera frame was lost'))
                if payload is not None:
                    yield payload
                return

            frame = cv2.flip(frame, 1)
            annotated_frame, _ = engine.process(frame)
            payload = encode_mjpeg_frame(annotated_frame)
            if payload is not None:
                yield payload
    finally:
        capture.release()

@app.route('/emoji')
def emoji():
    return render_template('emoji.html')

@app.route('/emoji_main')
def emoji_main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("going to generate frames function")
    return Response(
        generate_emoji(cap, gesture_engine),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__=="__main__":
    print("inside main")
    app.run(threaded=True)
