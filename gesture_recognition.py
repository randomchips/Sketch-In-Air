from collections import defaultdict, deque
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
import time

import cv2
import mediapipe as mp
import numpy as np


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGER_JOINTS = {
    'index': (5, 6, 7, 8),
    'middle': (9, 10, 11, 12),
    'ring': (13, 14, 15, 16),
    'pinky': (17, 18, 19, 20),
}

CANNED_GESTURES = {
    'Closed_Fist': 'Closed fist',
    'Open_Palm': 'Open palm',
    'Pointing_Up': 'Pointing up',
    'Thumb_Down': 'Thumb down',
    'Thumb_Up': 'Thumb up',
    'Victory': 'Victory',
    'ILoveYou': 'I love you',
}

GESTURE_EMOJI_INDEX = {
    'Closed fist': 10,
    'Open palm': 4,
    'Pointing up': 1,
    'One finger': 1,
    'Victory': 2,
    'Two fingers': 2,
    'Okay': 3,
    'I love you': 5,
    'Call me': 6,
    'Thumb up': 8,
}


@dataclass(frozen=True)
class HandGesture:
    handedness: str
    gesture: str
    confidence: float | None
    finger_count: int
    finger_states: tuple[bool, bool, bool, bool, bool]
    bounds: tuple[int, int, int, int]
    landmark_count: int


def _landmark_point(landmark):
    return np.array((landmark.x, landmark.y, landmark.z), dtype=np.float32)


def _joint_angle(first, center, last):
    first_vector = _landmark_point(first) - _landmark_point(center)
    last_vector = _landmark_point(last) - _landmark_point(center)
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(last_vector)
    if denominator < 1e-6:
        return 0.0
    cosine = np.dot(first_vector, last_vector) / denominator
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _landmark_distance(first, second):
    return float(np.linalg.norm(_landmark_point(first) - _landmark_point(second)))


def finger_states(landmarks):
    states = {
        'thumb': (
            _joint_angle(landmarks[1], landmarks[2], landmarks[3]) > 135
            and _joint_angle(landmarks[2], landmarks[3], landmarks[4]) > 145
            and _landmark_distance(landmarks[0], landmarks[4])
            > _landmark_distance(landmarks[0], landmarks[3]) * 1.08
        )
    }

    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        states[name] = (
            _joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip]) > 150
            and _joint_angle(landmarks[pip], landmarks[dip], landmarks[tip]) > 150
            and _landmark_distance(landmarks[0], landmarks[tip])
            > _landmark_distance(landmarks[0], landmarks[pip]) * 1.05
        )

    return states


def classify_landmark_gesture(landmarks, states):
    palm_size = max(_landmark_distance(landmarks[0], landmarks[9]), 1e-6)
    thumb_index_distance = _landmark_distance(landmarks[4], landmarks[8])

    if (
        thumb_index_distance < palm_size * 0.38
        and states['middle']
        and states['ring']
        and states['pinky']
    ):
        return 'Okay'

    if (
        states['thumb']
        and states['pinky']
        and not states['index']
        and not states['middle']
        and not states['ring']
    ):
        return 'Call me'

    if (
        states['thumb']
        and states['index']
        and states['pinky']
        and not states['middle']
        and not states['ring']
    ):
        return 'I love you'

    extended = [name for name, is_extended in states.items() if is_extended]
    if not extended:
        return 'Closed fist'
    if len(extended) == 5:
        return 'Open palm'
    if extended == ['index']:
        return 'One finger'
    if extended == ['index', 'middle']:
        return 'Two fingers'
    return f'{len(extended)} fingers'


def _weighted_mode(values):
    scores = defaultdict(int)
    for weight, value in enumerate(values, start=1):
        scores[value] += weight
    return max(scores, key=scores.get)


def enhance_low_light(frame, threshold=75.0):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(grayscale.mean())
    if brightness >= threshold:
        return frame, brightness

    normalized_brightness = max(brightness / 255.0, 0.01)
    gamma = float(np.clip(
        math.log(0.45) / math.log(normalized_brightness),
        0.35,
        1.0,
    ))
    lookup_table = np.array([
        ((value / 255.0) ** gamma) * 255
        for value in range(256)
    ], dtype=np.uint8)
    corrected = cv2.LUT(frame, lookup_table)

    lab_frame = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    lightness, first_channel, second_channel = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, first_channel, second_channel)),
        cv2.COLOR_LAB2BGR,
    )
    return enhanced, brightness


def _blend_rgba(background, foreground):
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    foreground_color = foreground[:, :, :3].astype(np.float32)
    background_color = background.astype(np.float32)
    return np.uint8(
        (foreground_color * alpha) + (background_color * (1.0 - alpha))
    )


def _overlay_rgba(image, overlay_image, x_coordinate, y_coordinate, size):
    if overlay_image is None or size <= 0:
        return

    overlay_height, overlay_width = overlay_image.shape[:2]
    scale = min(size / overlay_width, size / overlay_height)
    resized_width = max(1, int(overlay_width * scale))
    resized_height = max(1, int(overlay_height * scale))
    resized = cv2.resize(
        overlay_image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    target_x = x_coordinate + (size - resized_width) // 2
    target_y = y_coordinate + (size - resized_height) // 2
    left = max(0, target_x)
    top = max(0, target_y)
    right = min(image.shape[1], target_x + resized_width)
    bottom = min(image.shape[0], target_y + resized_height)
    if left >= right or top >= bottom:
        return

    source_left = left - target_x
    source_top = top - target_y
    source_right = source_left + (right - left)
    source_bottom = source_top + (bottom - top)
    foreground = resized[source_top:source_bottom, source_left:source_right]
    image[top:bottom, left:right] = _blend_rgba(
        image[top:bottom, left:right],
        foreground,
    )


class GestureEngine:
    def __init__(self, model_path, emoji_directory, num_hands=2):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise RuntimeError(f'Gesture model not found: {model_path}')

        self._lock = Lock()
        self._last_timestamp = 0
        self._gesture_history = defaultdict(lambda: deque(maxlen=7))
        self._finger_history = defaultdict(lambda: deque(maxlen=7))
        self._missed_frames = 0
        self._closed = False
        self._emoji_assets = self._load_emoji_assets(Path(emoji_directory))

        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._recognizer = (
            mp.tasks.vision.GestureRecognizer.create_from_options(options)
        )

    @staticmethod
    def _load_emoji_assets(emoji_directory):
        assets_by_index = {}
        for emoji_index in set(GESTURE_EMOJI_INDEX.values()):
            emoji_path = emoji_directory / f'{emoji_index}.png'
            emoji = cv2.imread(str(emoji_path), cv2.IMREAD_UNCHANGED)
            if emoji is None or emoji.ndim != 3 or emoji.shape[2] != 4:
                raise RuntimeError(f'Invalid emoji asset: {emoji_path}')
            assets_by_index[emoji_index] = emoji

        assets = {
            gesture: assets_by_index[emoji_index]
            for gesture, emoji_index in GESTURE_EMOJI_INDEX.items()
        }
        assets['Thumb down'] = cv2.rotate(
            assets_by_index[8],
            cv2.ROTATE_180,
        )
        return assets

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._recognizer.close()
            self._closed = True

    def process(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError('Cannot recognize gestures in an empty frame')

        recognition_frame, brightness = enhance_low_light(frame)
        rgb_frame = cv2.cvtColor(recognition_frame, cv2.COLOR_BGR2RGB)
        media_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        with self._lock:
            if self._closed:
                raise RuntimeError('Gesture engine is closed')
            timestamp = max(
                time.monotonic_ns() // 1_000_000,
                self._last_timestamp + 1,
            )
            self._last_timestamp = timestamp
            recognition = self._recognizer.recognize_for_video(
                media_image,
                timestamp,
            )
            observations = self._build_observations(recognition, frame.shape)

        annotated_frame = recognition_frame.copy()
        self._draw_results(
            annotated_frame,
            recognition,
            observations,
            brightness,
        )
        return annotated_frame, observations

    def _build_observations(self, recognition, frame_shape):
        if not recognition.hand_landmarks:
            self._missed_frames += 1
            if self._missed_frames >= 3:
                self._gesture_history.clear()
                self._finger_history.clear()
            return []

        self._missed_frames = 0
        observations = []
        for index, landmarks in enumerate(recognition.hand_landmarks):
            handedness = 'Hand'
            if index < len(recognition.handedness) and recognition.handedness[index]:
                handedness = recognition.handedness[index][0].category_name or 'Hand'

            states = finger_states(landmarks)
            finger_count = sum(states.values())
            raw_gesture, confidence = self._select_gesture(
                recognition,
                index,
                landmarks,
                states,
            )

            self._gesture_history[handedness].append((raw_gesture, confidence))
            self._finger_history[handedness].append(finger_count)
            stable_gesture = _weighted_mode(
                [entry[0] for entry in self._gesture_history[handedness]]
            )
            stable_finger_count = _weighted_mode(self._finger_history[handedness])
            stable_confidence = self._stable_confidence(
                self._gesture_history[handedness],
                stable_gesture,
            )

            observations.append(
                HandGesture(
                    handedness=handedness,
                    gesture=stable_gesture,
                    confidence=stable_confidence,
                    finger_count=stable_finger_count,
                    finger_states=tuple(states.values()),
                    bounds=self._landmark_bounds(landmarks, frame_shape),
                    landmark_count=len(landmarks),
                )
            )
        return observations

    @staticmethod
    def _select_gesture(recognition, index, landmarks, states):
        if index < len(recognition.gestures) and recognition.gestures[index]:
            category = recognition.gestures[index][0]
            display_name = CANNED_GESTURES.get(category.category_name)
            if display_name is not None and category.score >= 0.45:
                return display_name, float(category.score)
        return classify_landmark_gesture(landmarks, states), None

    @staticmethod
    def _stable_confidence(history, stable_gesture):
        weighted_total = 0.0
        total_weight = 0
        for weight, (gesture, confidence) in enumerate(history, start=1):
            if gesture == stable_gesture and confidence is not None:
                weighted_total += confidence * weight
                total_weight += weight
        if total_weight == 0:
            return None
        return weighted_total / total_weight

    @staticmethod
    def _landmark_bounds(landmarks, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x_coordinates = [landmark.x * frame_width for landmark in landmarks]
        y_coordinates = [landmark.y * frame_height for landmark in landmarks]
        padding = max(10, int(min(frame_width, frame_height) * 0.025))
        left = max(0, int(min(x_coordinates)) - padding)
        top = max(0, int(min(y_coordinates)) - padding)
        right = min(frame_width - 1, int(max(x_coordinates)) + padding)
        bottom = min(frame_height - 1, int(max(y_coordinates)) + padding)
        return left, top, right, bottom

    def _draw_results(self, frame, recognition, observations, brightness):
        for landmarks in recognition.hand_landmarks:
            self._draw_landmarks(frame, landmarks)

        for index, observation in enumerate(observations):
            self._draw_hand_label(frame, observation)
            emoji = self._emoji_assets.get(observation.gesture)
            if emoji is not None:
                emoji_size = max(64, int(min(frame.shape[:2]) * 0.19))
                margin = max(10, int(emoji_size * 0.12))
                _overlay_rgba(
                    frame,
                    emoji,
                    frame.shape[1] - emoji_size - margin,
                    frame.shape[0] - ((index + 1) * emoji_size) - ((index + 1) * margin),
                    emoji_size,
                )

        self._draw_frame_status(frame, len(observations), brightness)

    @staticmethod
    def _draw_landmarks(frame, landmarks):
        frame_height, frame_width = frame.shape[:2]
        points = [
            (
                int(np.clip(landmark.x, 0.0, 1.0) * (frame_width - 1)),
                int(np.clip(landmark.y, 0.0, 1.0) * (frame_height - 1)),
            )
            for landmark in landmarks
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (52, 194, 159),
                2,
                cv2.LINE_AA,
            )

        for index, point in enumerate(points):
            radius = 5 if index in (4, 8, 12, 16, 20) else 3
            fill_color = (35, 92, 235) if radius == 5 else (245, 245, 245)
            cv2.circle(frame, point, radius, fill_color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, (24, 42, 49), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hand_label(frame, observation):
        left, top, right, bottom = observation.bounds
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (52, 194, 159),
            2,
            cv2.LINE_AA,
        )

        finger_word = 'FINGER' if observation.finger_count == 1 else 'FINGERS'
        finger_label = f'{observation.finger_count} {finger_word}'
        if observation.gesture.casefold() == finger_label.casefold():
            label = f'{observation.handedness.upper()} | {finger_label}'
        else:
            label = (
                f'{observation.handedness.upper()} | '
                f'{observation.gesture.upper()} | '
                f'{finger_label}'
            )
        if observation.confidence is not None:
            label += f' | {int(observation.confidence * 100)}%'

        font_scale = max(0.38, min(frame.shape[:2]) / 980)
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        panel_width = min(frame.shape[1] - 1, text_size[0] + 16)
        panel_height = text_size[1] + baseline + 12
        panel_left = int(np.clip(left, 0, max(0, frame.shape[1] - panel_width)))
        panel_top = top - panel_height if top >= panel_height + 6 else bottom + 4
        status_clearance = max(54, int(min(frame.shape[:2]) * 0.12))
        if panel_top < status_clearance:
            panel_top = bottom + 4
        panel_top = int(np.clip(panel_top, 0, max(0, frame.shape[0] - panel_height)))
        cv2.rectangle(
            frame,
            (panel_left, panel_top),
            (panel_left + panel_width, panel_top + panel_height),
            (27, 37, 43),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (panel_left + 8, panel_top + text_size[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_frame_status(frame, hand_count, brightness):
        if hand_count == 0:
            label = (
                'LOW LIGHT | SHOW YOUR HAND'
                if brightness < 35
                else 'SHOW ONE OR TWO HANDS'
            )
            color = (49, 57, 64)
        else:
            suffix = 'HAND' if hand_count == 1 else 'HANDS'
            label = f'{hand_count} {suffix} DETECTED'
            color = (36, 101, 92)

        font_scale = max(0.46, min(frame.shape[:2]) / 900)
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        margin = max(10, int(min(frame.shape[:2]) * 0.025))
        panel_width = text_size[0] + (margin * 2)
        panel_height = text_size[1] + baseline + margin
        cv2.rectangle(
            frame,
            (margin, margin),
            (margin + panel_width, margin + panel_height),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (margin * 2, margin + text_size[1] + baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )