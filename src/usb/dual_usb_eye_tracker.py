import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HIGHGUI_AVAILABLE = True


LEFT_EYE = {
    "outer": 33,
    "inner": 133,
    "top": 159,
    "bottom": 145,
    "iris": [468, 469, 470, 471, 472],
}
RIGHT_EYE = {
    "outer": 362,
    "inner": 263,
    "top": 386,
    "bottom": 374,
    "iris": [473, 474, 475, 476, 477],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi USB camera gaze tracking prototype."
    )
    parser.add_argument("--cam-left", type=int, default=1, help="Left camera index.")
    parser.add_argument("--cam-right", type=int, default=2, help="Right camera index.")
    parser.add_argument("--cam-third", type=int, default=0, help="Center/third camera index.")
    parser.add_argument("--cam-fourth", type=int, default=3, help="Fourth camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS request.")
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="Camera pixel format request, for example MJPG or YUY2. MJPG lowers USB bandwidth.",
    )
    parser.add_argument(
        "--manual-exposure",
        action="store_true",
        help="Try to disable auto exposure on DirectShow cameras.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Manual exposure value. Many Windows webcams use negative values like -5 or -6.",
    )
    parser.add_argument("--gain", type=float, default=None, help="Camera gain value.")
    parser.add_argument("--brightness", type=float, default=None, help="Camera brightness value.")
    parser.add_argument("--contrast", type=float, default=None, help="Camera contrast value.")
    parser.add_argument(
        "--manual-white-balance",
        action="store_true",
        help="Try to disable auto white balance.",
    )
    parser.add_argument(
        "--white-balance",
        type=float,
        default=None,
        help="White balance temperature, for example 4500 or 5000.",
    )
    parser.add_argument(
        "--normalize-brightness",
        action="store_true",
        help="Software-normalize each frame brightness to reduce exposure flicker.",
    )
    parser.add_argument(
        "--target-brightness",
        type=float,
        default=115.0,
        help="Target grayscale mean for --normalize-brightness.",
    )
    parser.add_argument(
        "--brightness-alpha",
        type=float,
        default=0.15,
        help="Brightness normalization smoothing. Lower is steadier.",
    )
    parser.add_argument(
        "--use-cameras",
        choices=[
            "all", "both",
            "left", "right", "third", "fourth",
            "left-right", "left-third", "left-fourth",
            "right-third", "right-fourth", "third-fourth",
            "left-right-third", "left-right-fourth",
            "left-third-fourth", "right-third-fourth",
        ],
        default="all",
        help="Choose which configured cameras to use.",
    )
    parser.add_argument(
        "--camera-indexes",
        default=None,
        help="Comma-separated camera indexes. Overrides --cam-left/--cam-right/--cam-third/--cam-fourth.",
    )
    parser.add_argument(
        "--monitor-camera-indexes",
        default=None,
        help="Extra camera indexes to show as previews, but not use for gaze calibration.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List camera indexes.")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only show raw camera previews without calibration or eye detection.",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save one frame from each selected camera, then exit.",
    )
    parser.add_argument(
        "--diagnose-cameras",
        action="store_true",
        help="Show raw previews and print per-camera FPS/read-failure diagnostics.",
    )
    parser.add_argument("--calibration", default="calibration.json", help="Save path.")
    parser.add_argument("--load-calibration", default=None, help="Load calibration JSON.")
    parser.add_argument(
        "--cubic",
        action="store_true",
        help="Use cubic polynomial calibration for better non-linear gaze mapping.",
    )
    parser.add_argument(
        "--linear-calibration",
        action="store_true",
        help="Use a linear gaze mapping instead of the default quadratic mapping.",
    )
    parser.add_argument("--screen-width", type=int, default=None)
    parser.add_argument("--screen-height", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=["auto", "mediapipe", "haar"],
        default="auto",
        help="Eye feature backend. auto uses MediaPipe when available, then Haar fallback.",
    )
    parser.add_argument(
        "--mp-no-head-pose",
        action="store_true",
        help="Disable head pose estimation in MediaPipe backend (reduces features from 7 to 4 per camera).",
    )
    parser.add_argument(
        "--haar-face-min-neighbors",
        type=int,
        default=4,
        help="Haar face detection minNeighbors. Lower = more detections but more false positives (default 4).",
    )
    parser.add_argument(
        "--haar-eye-min-neighbors",
        type=int,
        default=5,
        help="Haar eye detection minNeighbors. Lower = more detections but more false positives (default 5).",
    )
    parser.add_argument(
        "--haar-face-scale",
        type=float,
        default=1.1,
        help="Haar face detection scaleFactor (default 1.1).",
    )
    parser.add_argument(
        "--haar-eye-scale",
        type=float,
        default=1.08,
        help="Haar eye detection scaleFactor (default 1.08).",
    )
    parser.add_argument(
        "--haar-face-min-size",
        type=int,
        default=100,
        help="Haar face detection minSize in pixels (default 100).",
    )
    parser.add_argument("--samples-per-point", type=int, default=45)
    parser.add_argument(
        "--calibration-rounds",
        type=int,
        default=1,
        help="How many times to repeat the 9-point calibration grid.",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument(
        "--feature-alpha",
        type=float,
        default=0.35,
        help="Feature smoothing amount from 0 to 1. Lower is smoother.",
    )
    parser.add_argument(
        "--max-feature-jump",
        type=float,
        default=0.22,
        help="Reject per-camera feature jumps larger than this normalized distance.",
    )
    parser.add_argument(
        "--gaze-alpha",
        type=float,
        default=0.22,
        help="Gaze point smoothing amount from 0 to 1. Lower is steadier, higher is faster.",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror camera previews.")
    parser.add_argument(
        "--no-hold-last-frame",
        action="store_true",
        help="Do not reuse the last good frame when a camera read fails.",
    )
    parser.add_argument("--windowed", action="store_true", help="Do not use fullscreen.")
    parser.add_argument(
        "--embed-previews",
        action="store_true",
        default=True,
        help="Embed camera previews into the gaze/calibration window.",
    )
    parser.add_argument(
        "--separate-previews",
        action="store_false",
        dest="embed_previews",
        help="Show camera previews in a separate window.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After calibration, run a validation phase with known target points to measure accuracy.",
    )
    parser.add_argument(
        "--validate-points",
        default=None,
        help="Comma-separated validation point coordinates in normalized screen coords, e.g. '0.25,0.30,0.75,0.30,0.50,0.50,0.25,0.70,0.75,0.70'.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=100,
        help="Samples to collect per validation point (default 100).",
    )
    parser.add_argument(
        "--validate-settle",
        type=float,
        default=0.8,
        help="Settle time in seconds before sampling each validation point.",
    )
    parser.add_argument(
        "--validate-output",
        default=None,
        help="Optional path to save validation raw data JSON.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark comparing multiple camera configurations.",
    )
    parser.add_argument(
        "--benchmark-configs",
        default="left,all",
        help="Comma-separated --use-cameras values to compare, e.g. 'left,left-right,all'.",
    )
    parser.add_argument(
        "--benchmark-trials",
        type=int,
        default=2,
        help="Number of trials per configuration (default 2).",
    )
    parser.add_argument(
        "--benchmark-output",
        default="benchmark_result.json",
        help="Output path for benchmark results.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run statistical analysis on previously collected validation data files.",
    )
    parser.add_argument(
        "--analyze-files",
        default=None,
        help="Comma-separated list of validation raw data JSON files to analyze.",
    )
    parser.add_argument(
        "--latency-test",
        action="store_true",
        help="Measure system latency by detecting gaze response delay after stimulus jumps.",
    )
    parser.add_argument(
        "--latency-trials",
        type=int,
        default=5,
        help="Number of stimulus jumps for latency test (default 5).",
    )
    parser.add_argument(
        "--drift-test",
        action="store_true",
        help="Measure gaze prediction drift over time while fixating a single point.",
    )
    parser.add_argument(
        "--drift-duration",
        type=float,
        default=30.0,
        help="Drift test duration in seconds (default 30).",
    )
    # --- Algorithm improvement flags (all default off for backward compatibility) ---
    parser.add_argument(
        "--partial-fusion",
        action="store_true",
        help="Allow gaze prediction when only a subset of cameras detect eyes. Averages per-camera model predictions.",
    )
    parser.add_argument(
        "--min-active-cameras",
        type=int,
        default=1,
        help="Minimum detected cameras required for --partial-fusion calibration/prediction samples.",
    )
    parser.add_argument(
        "--adaptive-ridge",
        action="store_true",
        help="Auto-scale Ridge lambda based on feature dimensionality vs sample count.",
    )
    parser.add_argument(
        "--robust-fit",
        action="store_true",
        help="Use Huber loss (IRLS) instead of least-squares for calibration to reject blink/head-move outliers.",
    )
    parser.add_argument(
        "--adaptive-pupil",
        action="store_true",
        help="Multi-threshold pupil detection with circularity scoring, corneal reflection inpainting, and ellipse fitting (Haar backend only).",
    )
    parser.add_argument(
        "--one-euro-filter",
        action="store_true",
        help="Replace velocity-adaptive EMA with 1-euro filter for lower-lag gaze smoothing.",
    )
    parser.add_argument(
        "--euro-min-cutoff",
        type=float,
        default=1.0,
        help="1-euro filter minimum cutoff frequency in Hz (default 1.0).",
    )
    parser.add_argument(
        "--euro-beta",
        type=float,
        default=0.007,
        help="1-euro filter speed coefficient (default 0.007).",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=None,
        help="Override Ridge regression lambda. When set, --adaptive-ridge is ignored. Typical: 0.005-0.05 for 4-cam, 0.001-0.01 for 1-cam.",
    )
    parser.add_argument(
        "--threaded-reads",
        action="store_true",
        help="Read all cameras in parallel threads to reduce latency.",
    )
    return parser.parse_args()


def get_screen_size(args):
    if args.screen_width and args.screen_height:
        return args.screen_width, args.screen_height
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        return 1280, 720


def list_cameras(max_index=10):
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
    return found


def set_camera_property(cap, prop, value, label):
    if value is None:
        return
    ok = cap.set(prop, float(value))
    actual = cap.get(prop)
    print(f"  {label}: requested={value}, actual={actual}, set_ok={ok}")


def open_camera(index, args):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if args.fourcc:
        code = str(args.fourcc).upper()[:4]
        if len(code) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.manual_exposure:
        set_camera_property(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure")
    set_camera_property(cap, cv2.CAP_PROP_EXPOSURE, args.exposure, "exposure")
    set_camera_property(cap, cv2.CAP_PROP_GAIN, args.gain, "gain")
    set_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, args.brightness, "brightness")
    set_camera_property(cap, cv2.CAP_PROP_CONTRAST, args.contrast, "contrast")
    if args.manual_white_balance:
        set_camera_property(cap, cv2.CAP_PROP_AUTO_WB, 0, "auto_white_balance")
    set_camera_property(cap, cv2.CAP_PROP_WB_TEMPERATURE, args.white_balance, "white_balance")
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {index}.")
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    chars = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(
        f"camera {index}: size={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}, fourcc={chars!r}"
    )
    return cap


class CameraStream:
    def __init__(
        self,
        index,
        cap,
        hold_last_frame=True,
        normalize_brightness=False,
        target_brightness=115.0,
        brightness_alpha=0.15,
    ):
        self.index = index
        self.cap = cap
        self.hold_last_frame = hold_last_frame
        self.normalize_brightness = normalize_brightness
        self.target_brightness = float(target_brightness)
        self.brightness_alpha = float(np.clip(brightness_alpha, 0.01, 1.0))
        self.brightness_gain = 1.0
        self.last_frame = None
        self.reads = 0
        self.failures = 0
        self.reused = 0
        self.started_at = time.time()

    def read(self):
        ok, frame = self.cap.read()
        self.reads += 1
        if ok and frame is not None:
            frame = self._normalize_brightness(frame)
            self.last_frame = frame
            return True, frame, False
        self.failures += 1
        if self.hold_last_frame and self.last_frame is not None:
            self.reused += 1
            return True, self.last_frame.copy(), True
        return False, None, False

    def _normalize_brightness(self, frame):
        if not self.normalize_brightness:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        if mean < 1.0:
            return frame
        desired_gain = float(np.clip(self.target_brightness / mean, 0.45, 2.25))
        self.brightness_gain = (
            (1.0 - self.brightness_alpha) * self.brightness_gain
            + self.brightness_alpha * desired_gain
        )
        return cv2.convertScaleAbs(frame, alpha=self.brightness_gain, beta=0)

    def stats(self):
        elapsed = max(time.time() - self.started_at, 0.001)
        return {
            "index": self.index,
            "fps": self.reads / elapsed,
            "reads": self.reads,
            "failures": self.failures,
            "reused": self.reused,
        }

    def release(self):
        self.cap.release()


def draw_label(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def selected_camera_indexes(args):
    if args.camera_indexes:
        indexes = []
        for raw in args.camera_indexes.split(","):
            raw = raw.strip()
            if raw:
                indexes.append(int(raw))
        if not indexes:
            raise SystemExit("--camera-indexes did not contain any indexes.")
        return indexes

    named = {
        "left": args.cam_left,
        "right": args.cam_right,
        "third": args.cam_third,
        "fourth": args.cam_fourth,
    }
    groups = {
        "all": ["left", "right", "third", "fourth"],
        "both": ["left", "right"],
        "left-right": ["left", "right"],
        "left-third": ["left", "third"],
        "left-fourth": ["left", "fourth"],
        "right-third": ["right", "third"],
        "right-fourth": ["right", "fourth"],
        "third-fourth": ["third", "fourth"],
        "left-right-third": ["left", "right", "third"],
        "left-right-fourth": ["left", "right", "fourth"],
        "left-third-fourth": ["left", "third", "fourth"],
        "right-third-fourth": ["right", "third", "fourth"],
        "left": ["left"],
        "right": ["right"],
        "third": ["third"],
        "fourth": ["fourth"],
    }
    return [named[name] for name in groups[args.use_cameras]]


def parse_index_list(raw):
    if not raw:
        return []
    indexes = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            indexes.append(int(item))
    return indexes


def require_gaze_cameras(camera_indexes, args):
    if args.list_cameras or args.preview_only or args.diagnose_cameras or args.save_snapshots:
        return
    if len(camera_indexes) < 1:
        raise SystemExit(
            "At least one gaze-control camera is required. "
            f"Selected {len(camera_indexes)} camera(s): {camera_indexes}. "
            "Use --use-cameras or --camera-indexes to select cameras."
        )


def lm_xy(landmarks, idx, width, height):
    lm = landmarks[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def eye_feature(landmarks, width, height, eye):
    outer = lm_xy(landmarks, eye["outer"], width, height)
    inner = lm_xy(landmarks, eye["inner"], width, height)
    top = lm_xy(landmarks, eye["top"], width, height)
    bottom = lm_xy(landmarks, eye["bottom"], width, height)
    iris = np.mean([lm_xy(landmarks, idx, width, height) for idx in eye["iris"]], axis=0)

    center = (outer + inner + top + bottom) / 4.0
    eye_w = max(float(np.linalg.norm(inner - outer)), 1.0)
    eye_h = max(float(np.linalg.norm(bottom - top)), 1.0)
    rel = iris - center
    return np.array([rel[0] / eye_w, rel[1] / eye_h], dtype=np.float32)


class MediaPipeEyeExtractor:
    # Simplified 3D face model (mm scale) for head pose estimation
    _FACE_MODEL = np.array([
        (0.0, 0.0, 0.0),        # Nose tip (landmark 1)
        (0.0, -63.6, -12.5),    # Chin (landmark 152)
        (-43.3, 32.7, -26.0),   # Left eye outer corner (landmark 33)
        (43.3, 32.7, -26.0),    # Right eye outer corner (landmark 362)
        (-28.9, -28.9, -24.1),  # Left mouth corner (landmark 61)
        (28.9, -28.9, -24.1),   # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    _HEAD_POSE_IDXS = [1, 152, 33, 362, 61, 291]

    FEATURE_COUNT = 7  # 4 eye features + 3 head pose (rotation vector)

    def __init__(self, estimate_head_pose=True):
        self.face_mesh_module = load_mediapipe_face_mesh()
        self.model = self.face_mesh_module.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.estimate_head_pose = estimate_head_pose
        if not estimate_head_pose:
            self.FEATURE_COUNT = 4  # Only eye features

    def process_frame(self, frame, mirror=False):
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.process(rgb)
        if not result.multi_face_landmarks:
            return None, frame

        h, w = frame.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark
        left = eye_feature(landmarks, w, h, LEFT_EYE)
        right = eye_feature(landmarks, w, h, RIGHT_EYE)
        feat_parts = [left, right]

        # Head pose estimation
        head_rvec = None
        if self.estimate_head_pose:
            head_rvec = self._estimate_head_pose(landmarks, w, h)
            feat_parts.append(head_rvec)

        feat = np.concatenate(feat_parts)

        # Visualization
        for eye in (LEFT_EYE, RIGHT_EYE):
            iris = np.mean([lm_xy(landmarks, idx, w, h) for idx in eye["iris"]], axis=0)
            cv2.circle(frame, tuple(iris.astype(int)), 3, (0, 255, 255), -1)
        if head_rvec is not None:
            self._draw_head_pose(frame, landmarks, w, h, head_rvec)
        return feat, frame

    def _estimate_head_pose(self, landmarks, w, h):
        """Estimate head rotation from 6 facial landmarks via PnP."""
        image_points = np.array([
            lm_xy(landmarks, idx, w, h) for idx in self._HEAD_POSE_IDXS
        ], dtype=np.float64)

        focal = float(w)
        camera_matrix = np.array([
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, _ = cv2.solvePnP(
                self._FACE_MODEL, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                return rvec.flatten().astype(np.float32)
        except cv2.error:
            pass
        return np.zeros(3, dtype=np.float32)

    def _draw_head_pose(self, frame, landmarks, w, h, rvec):
        """Draw a head-pose axis at the nose tip."""
        nose_xy = lm_xy(landmarks, 1, w, h).astype(int)

        focal = float(w)
        camera_matrix = np.array([
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)  # approximate

        axis_len = int(min(w, h) * 0.12)
        axis_3d = np.array([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ], dtype=np.float64)

        try:
            img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            cv2.line(frame, tuple(nose_xy), tuple(img_pts[0]), (0, 0, 255), 2)  # X red
            cv2.line(frame, tuple(nose_xy), tuple(img_pts[1]), (0, 255, 0), 2)  # Y green
            cv2.line(frame, tuple(nose_xy), tuple(img_pts[2]), (255, 0, 0), 2)  # Z blue
        except cv2.error:
            pass

    def close(self):
        self.model.close()


class HaarEyeExtractor:
    FEATURE_COUNT = 4  # 2 normalized pupil coords per eye

    def __init__(self, face_min_neighbors=4, eye_min_neighbors=5,
                 face_scale=1.1, eye_scale=1.08, face_min_size=100,
                 adaptive_pupil=False):
        base = Path(cv2.data.haarcascades)
        self.face = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        self.eye = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
        # Glasses-friendly cascade as extra fallback
        glasses_path = str(base / "haarcascade_eye_tree_eyeglasses.xml")
        self.eye_glasses = None
        if Path(glasses_path).exists():
            self.eye_glasses = cv2.CascadeClassifier(glasses_path)
        if self.face.empty() or self.eye.empty():
            raise SystemExit("OpenCV Haar cascade files were not found.")
        self.face_min_neighbors = face_min_neighbors
        self.eye_min_neighbors = eye_min_neighbors
        self.face_scale = face_scale
        self.eye_scale = eye_scale
        self.face_min_size = face_min_size
        self.adaptive_pupil = adaptive_pupil

    def process_frame(self, frame, mirror=False):
        if mirror:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face.detectMultiScale(
            gray, self.face_scale, self.face_min_neighbors,
            minSize=(self.face_min_size, self.face_min_size),
        )
        if len(faces) == 0:
            return None, frame

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_gray = gray[y : y + h, x : x + w]
        face_frame = frame[y : y + h, x : x + w]

        # Multi-pass eye detection
        eye_candidates = self.eye.detectMultiScale(
            face_gray, self.eye_scale, self.eye_min_neighbors,
            minSize=(24, 14),
            maxSize=(max(w // 2, 30), max(h // 3, 24)),
        )

        # Glasses cascade as first fallback
        if len(eye_candidates) == 0 and self.eye_glasses is not None:
            eye_candidates = self.eye_glasses.detectMultiScale(
                face_gray, self.eye_scale, self.eye_min_neighbors,
                minSize=(24, 14),
                maxSize=(max(w // 2, 30), max(h // 3, 24)),
            )

        # Relaxed final pass
        if len(eye_candidates) == 0:
            eye_candidates = self.eye.detectMultiScale(
                face_gray, 1.05, 3,
                minSize=(20, 12),
                maxSize=(max(w // 2, 30), max(h // 3, 24)),
            )
            if eye_candidates is not None and len(eye_candidates) > 0:
                eye_candidates = [e for e in eye_candidates if e[1] <= h * 0.65]
            else:
                eye_candidates = []

        eyes = []
        for ex, ey, ew, eh in eye_candidates:
            if ey > h * 0.62:
                continue
            eyes.append((ex, ey, ew, eh))
        if len(eyes) < 2:
            return None, frame

        eyes = sorted(eyes, key=lambda rect: rect[2] * rect[3], reverse=True)[:2]
        eyes = sorted(eyes, key=lambda rect: rect[0])
        features = []
        for ex, ey, ew, eh in eyes:
            roi = face_gray[ey : ey + eh, ex : ex + ew]
            pupil = None
            if self.adaptive_pupil:
                result = self._pupil_center_adaptive(roi)
                if result is not None:
                    pupil = (result[0], result[1])  # cx, cy (ignore quality score)
            if pupil is None:
                pupil = self._pupil_center(roi)
            if pupil is None:
                pupil = self._pupil_center_darkest(roi)
            if pupil is None:
                pupil = (ew / 2.0, eh / 2.0)
            px, py = pupil
            features.extend([(px - ew / 2.0) / max(ew, 1), (py - eh / 2.0) / max(eh, 1)])
            center = (x + ex + int(px), y + ey + int(py))
            cv2.rectangle(face_frame, (ex, ey), (ex + ew, ey + eh), (0, 180, 255), 1)
            cv2.circle(frame, center, 3, (0, 255, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 180, 80), 1)
        return np.asarray(features, dtype=np.float32), frame

    def _pupil_center(self, roi):
        if roi.size == 0:
            return None
        blur = cv2.GaussianBlur(roi, (7, 7), 0)
        cutoff = np.percentile(blur, 12)
        mask = (blur <= cutoff).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return None
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    def _pupil_center_darkest(self, roi):
        """Fallback: find the darkest pixel as pupil estimate."""
        if roi.size == 0:
            return None
        blur = cv2.GaussianBlur(roi, (9, 9), 0)
        min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
        mean_val = np.mean(blur)
        if min_val < mean_val * 0.7:
            return (float(min_loc[0]), float(min_loc[1]))
        return None

    def _pupil_center_adaptive(self, roi):
        """Multi-threshold pupil detection with quality scoring and ellipse fitting.

        Tries percentile thresholds [5, 8, 12, 18, 25], inpaints corneal reflections,
        scores each candidate by circularity + size, fits ellipse for sub-pixel center.
        Returns (cx, cy, quality) or None.
        """
        if roi.size == 0:
            return None

        blur = cv2.GaussianBlur(roi, (7, 7), 0)

        # Corneal reflection inpainting
        bright_cutoff = np.percentile(blur, 85)
        reflection_mask = (blur >= bright_cutoff).astype(np.uint8) * 255
        inpainted = cv2.inpaint(blur, reflection_mask, 3, cv2.INPAINT_TELEA)

        best_score = -1.0
        best_center = None

        for pct in [5, 8, 12, 18, 25]:
            cutoff = np.percentile(inpainted, pct)
            mask = (inpainted <= cutoff).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            roi_area = roi.shape[0] * roi.shape[1]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < roi_area * 0.02 or area > roi_area * 0.50:
                    continue
                if len(contour) < 5:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter < 1.0:
                    continue

                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                area_ratio = area / roi_area
                size_score = 1.0 - abs(area_ratio - 0.15) / 0.15
                size_score = max(0.0, size_score)
                score = circularity * 0.7 + size_score * 0.3

                if score > best_score:
                    best_score = score
                    ellipse = cv2.fitEllipse(contour)
                    best_center = (float(ellipse[0][0]), float(ellipse[0][1]), score)

        if best_center is not None:
            return best_center
        return None

    def close(self):
        return None


class OneEuroFilter:
    """1-euro filter for low-lag signal smoothing.

    From: Casiez, Roussel, Vogel. "1 Euro Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems." CHI 2012.

    Parameters:
      min_cutoff: minimum cutoff frequency in Hz (lower = smoother but more lag).
                  Default 1.0 works well for gaze at 30Hz.
      beta: speed coefficient. Higher = less lag during fast movement.
            Default 0.007 is typical for pointing tasks.
      d_cutoff: derivative cutoff frequency in Hz. Default 1.0.
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        self.initialized = False

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-8))
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        if not self.initialized:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            self.initialized = True
            return self.x_prev

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0 / 30.0

        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = (1.0 - alpha_d) * self.dx_prev + alpha_d * dx

        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)
        x_hat = (1.0 - alpha) * self.x_prev + alpha * x

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        self.initialized = False


class FeatureStabilizer:
    def __init__(self, alpha, max_jump):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.max_jump = max(float(max_jump), 0.0)
        self.value = None
        self.rejected = 0

    def update(self, feature):
        feature = np.asarray(feature, dtype=np.float32)
        if self.value is None:
            self.value = feature
            return self.value

        jump = float(np.linalg.norm(feature - self.value))
        if self.max_jump > 0 and jump > self.max_jump:
            self.rejected += 1
            return self.value

        self.rejected = 0
        self.value = (1.0 - self.alpha) * self.value + self.alpha * feature
        return self.value


def load_mediapipe_face_mesh():
    if sys.version_info >= (3, 13):
        raise ImportError(
            "MediaPipe's classic FaceMesh API is usually unavailable on Python 3.13+."
        )
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return mp.solutions.face_mesh
    except Exception:
        pass
    try:
        from mediapipe.python.solutions import face_mesh

        return face_mesh
    except Exception as exc:
        raise ImportError("MediaPipe FaceMesh API is unavailable.") from exc


def create_extractors(backend, count, haar_face_min_neighbors=4,
                      haar_eye_min_neighbors=5, haar_face_scale=1.1,
                      haar_eye_scale=1.08, haar_face_min_size=100,
                      mp_no_head_pose=False, adaptive_pupil=False):
    if backend in ("auto", "mediapipe"):
        try:
            extractors = [MediaPipeEyeExtractor(
                estimate_head_pose=not mp_no_head_pose
            ) for _ in range(count)]
            print(f"Using MediaPipe FaceMesh backend (features={extractors[0].FEATURE_COUNT}/cam).")
            return extractors
        except Exception as exc:
            if backend == "mediapipe":
                raise SystemExit(
                    "MediaPipe backend is not available. On Windows, use Python 3.10-3.12 "
                    "or run with --backend haar.\n"
                    f"Reason: {exc}"
                ) from exc
            print(f"MediaPipe unavailable, falling back to Haar backend: {exc}")
    print("Using OpenCV Haar backend.")
    return [HaarEyeExtractor(
        face_min_neighbors=haar_face_min_neighbors,
        eye_min_neighbors=haar_eye_min_neighbors,
        face_scale=haar_face_scale,
        eye_scale=haar_eye_scale,
        face_min_size=haar_face_min_size,
        adaptive_pupil=adaptive_pupil,
    ) for _ in range(count)]


def poly_features(x, cubic=False, linear=False):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    parts = [np.ones(1, dtype=np.float32), x]
    if linear:
        return np.concatenate(parts)
    # Quadratic terms: x_i * x_j for i <= j
    quad = [x[i] * x[j] for i in range(n) for j in range(i, n)]
    parts.append(np.asarray(quad, dtype=np.float32))
    # Cubic terms: x_i * x_j * x_k for i <= j <= k
    if cubic and n > 0:
        cube = []
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    cube.append(x[i] * x[j] * x[k])
        parts.append(np.asarray(cube, dtype=np.float32))
    return np.concatenate(parts)


def poly_feature_count(raw_feature_count, cubic=False, linear=False):
    n = raw_feature_count
    if linear:
        return 1 + n
    count = 1 + n + n * (n + 1) // 2  # constant + linear + quadratic
    if cubic and n > 0:
        # Number of cubic terms = C(n+2,3)
        count += n * (n + 1) * (n + 2) // 6
    return count


def fit_ridge(features, targets, lam=1e-2, cubic=False, adaptive_lam=False, linear=False):
    x = np.vstack([poly_features(row, cubic=cubic, linear=linear) for row in features])
    y = np.asarray(targets, dtype=np.float32)
    n_samples, n_features = x.shape

    if adaptive_lam:
        ratio = n_samples / max(n_features, 1)
        if ratio < 10.0:
            lam = float(np.clip((10.0 - ratio) / 10.0 * 0.5 + 1e-4, 1e-4, 1.0))
            print(f"  Adaptive lambda: {lam:.4f} (n_samples={n_samples}, n_features={n_features}, ratio={ratio:.1f})")

    eye = np.eye(x.shape[1], dtype=np.float32)
    eye[0, 0] = 0.0
    # Use more stable solver with fallback
    try:
        weights = np.linalg.solve(x.T @ x + lam * eye, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(x.T @ x + lam * eye, x.T @ y, rcond=None)[0]
    return weights


def fit_ridge_huber(features, targets, lam=1e-2, cubic=False, adaptive_lam=False,
                     linear=False,
                     max_iter=20, tol=1e-4):
    """Fit Ridge regression with Huber loss via IRLS.

    Huber delta is set to 1.345 * MAD(residuals) for ~95% efficiency
    under normality while protecting against blink/head-move outliers.
    """
    x = np.vstack([poly_features(row, cubic=cubic, linear=linear) for row in features])
    y = np.asarray(targets, dtype=np.float32)
    n_samples, n_features = x.shape

    if adaptive_lam:
        ratio = n_samples / max(n_features, 1)
        if ratio < 10.0:
            lam = float(np.clip((10.0 - ratio) / 10.0 * 0.5 + 1e-4, 1e-4, 1.0))

    eye = np.eye(n_features, dtype=np.float32)
    eye[0, 0] = 0.0

    try:
        weights = np.linalg.solve(x.T @ x + lam * eye, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(x.T @ x + lam * eye, x.T @ y, rcond=None)[0]

    for _ in range(max_iter):
        residuals = y - x @ weights
        huber_w = np.ones(n_samples, dtype=np.float32)
        for dim in range(y.shape[1]):
            r_dim = residuals[:, dim]
            mad = float(np.median(np.abs(r_dim - np.median(r_dim))))
            if mad < 1e-8:
                continue
            delta = 1.345 * mad
            abs_r = np.abs(r_dim)
            w_dim = np.where(abs_r <= delta, 1.0, delta / abs_r)
            huber_w = np.minimum(huber_w, w_dim.astype(np.float32))

        w_sqrt = np.sqrt(huber_w)
        xw = x * w_sqrt[:, np.newaxis]
        yw = y * w_sqrt[:, np.newaxis]
        try:
            new_weights = np.linalg.solve(xw.T @ xw + lam * eye, xw.T @ yw)
        except np.linalg.LinAlgError:
            new_weights = np.linalg.lstsq(xw.T @ xw + lam * eye, xw.T @ yw, rcond=None)[0]

        change = float(np.linalg.norm(new_weights - weights))
        weights = new_weights
        if change < tol:
            break

    return weights


def calibration_fit_metrics(model, samples, targets, cubic=False, linear=False, active_masks=None):
    """Measure how well a freshly fitted model explains its calibration samples."""
    preds = []
    if active_masks is None:
        active_masks = [None] * len(samples)
    for sample, active_mask in zip(samples, active_masks):
        pred, _source = _predict_dispatch(model, sample, active_mask, cubic=cubic, linear=linear)
        preds.append(pred)
    pred_arr = np.asarray(preds, dtype=np.float32)
    target_arr = np.asarray(targets, dtype=np.float32)
    errors = np.linalg.norm(pred_arr - target_arr, axis=1)
    span_x = float(np.ptp(pred_arr[:, 0])) if len(pred_arr) else 0.0
    span_y = float(np.ptp(pred_arr[:, 1])) if len(pred_arr) else 0.0
    return {
        "rmse_px": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("inf"),
        "mean_error_px": float(np.mean(errors)) if len(errors) else float("inf"),
        "p95_error_px": float(np.percentile(errors, 95)) if len(errors) else float("inf"),
        "span_x": span_x,
        "span_y": span_y,
    }


def validate_calibration_fit(model, samples, targets, screen_w, screen_h, cubic=False, linear=False, active_masks=None):
    """Reject collapsed or clearly broken calibration fits before saving them."""
    metrics = calibration_fit_metrics(
        model, samples, targets, cubic=cubic, linear=linear, active_masks=active_masks
    )
    min_span_x = screen_w * 0.20
    min_span_y = screen_h * 0.20
    max_rmse = max(screen_w, screen_h) * 0.28
    print(
        "  Calibration fit check: "
        f"rmse={metrics['rmse_px']:.1f}px, "
        f"mean={metrics['mean_error_px']:.1f}px, "
        f"p95={metrics['p95_error_px']:.1f}px, "
        f"span=({metrics['span_x']:.1f},{metrics['span_y']:.1f})"
    )
    if metrics["span_x"] < min_span_x or metrics["span_y"] < min_span_y:
        raise SystemExit(
            "Calibration fit collapsed: predictions do not cover enough of the screen. "
            "Check eye detection, lighting, and camera placement, then recalibrate."
        )
    if metrics["rmse_px"] > max_rmse:
        raise SystemExit(
            "Calibration fit error is too high. "
            "Keep looking at each dot during calibration or increase --samples-per-point."
        )
    return metrics


def predict(weights, feature, cubic=False, linear=False):
    return poly_features(feature, cubic=cubic, linear=linear) @ weights


class MultiCameraRidgeModel:
    """Per-camera Ridge models + optional full fusion model for partial-fusion.

    Serialization v2 format:
      {"version": 2, "camera_count": N, "features_per_camera": F, "cubic": bool,
       "linear": bool, "full_weights": [...], "per_camera_weights": [[...], ...]}

    Legacy v1 format (no "version" key) loads as full_weights only.
    """

    def __init__(self, camera_count, features_per_camera, cubic=False, linear=False):
        self.camera_count = camera_count
        self.features_per_camera = features_per_camera
        self.cubic = cubic
        self.linear = linear
        self.full_weights = None
        self.per_camera_weights = []

    @property
    def has_full_model(self):
        return self.full_weights is not None

    @property
    def has_per_camera_models(self):
        return (
            len(self.per_camera_weights) == self.camera_count
            and any(w is not None for w in self.per_camera_weights)
        )

    def fit_full(self, samples, targets, lam=1e-2, robust=False):
        fitter = fit_ridge_huber if robust else fit_ridge
        self.full_weights = fitter(samples, targets, lam=lam, cubic=self.cubic, linear=self.linear)

    def fit_per_camera(self, samples, targets, lam=1e-2, robust=False):
        fitter = fit_ridge_huber if robust else fit_ridge
        fpc = self.features_per_camera
        self.per_camera_weights = []
        for cam_idx in range(self.camera_count):
            start = cam_idx * fpc
            end = start + fpc
            cam_samples = [s[start:end] for s in samples]
            w = fitter(cam_samples, targets, lam=lam, cubic=self.cubic, linear=self.linear)
            self.per_camera_weights.append(w)

    def predict(self, feature, active_mask=None):
        """Predict gaze point.

        active_mask=None or all True → full model.
        Partial cameras active → average per-camera predictions.
        Returns (predicted_xy, source_string).
        """
        if active_mask is None:
            active_mask = [True] * self.camera_count
        else:
            active_mask = list(active_mask)

        if all(active_mask):
            if self.full_weights is not None:
                return predict(self.full_weights, feature, cubic=self.cubic, linear=self.linear), "full"
            elif self.has_per_camera_models:
                pass  # fall through to per-camera averaging
            else:
                return np.array([0.0, 0.0], dtype=np.float32), "none"

        if not self.has_per_camera_models:
            if self.full_weights is not None:
                return predict(self.full_weights, feature, cubic=self.cubic, linear=self.linear), "full_zero_fill"
            return np.array([0.0, 0.0], dtype=np.float32), "none"

        fpc = self.features_per_camera
        predictions = []
        for cam_idx, active in enumerate(active_mask):
            if not active or cam_idx >= len(self.per_camera_weights):
                continue
            if self.per_camera_weights[cam_idx] is None:
                continue
            start = cam_idx * fpc
            end = start + fpc
            cam_feat = feature[start:end]
            pred = predict(
                self.per_camera_weights[cam_idx], cam_feat,
                cubic=self.cubic, linear=self.linear,
            )
            predictions.append(pred)

        if not predictions:
            return np.array([0.0, 0.0], dtype=np.float32), "none"
        avg = np.mean(predictions, axis=0)
        return avg.astype(np.float32), "partial_avg"

    def to_dict(self):
        d = {
            "version": 2,
            "camera_count": self.camera_count,
            "features_per_camera": self.features_per_camera,
            "cubic": self.cubic,
            "linear": self.linear,
        }
        if self.full_weights is not None:
            d["full_weights"] = self.full_weights.tolist()
        d["per_camera_weights"] = [
            w.tolist() if w is not None else None for w in self.per_camera_weights
        ]
        return d

    @classmethod
    def from_dict(cls, d):
        if "version" not in d or d.get("version", 0) < 2:
            # Legacy v1 format: single weights array
            obj = cls(
                camera_count=1,
                features_per_camera=0,
                cubic=d.get("cubic", False),
                linear=d.get("linear", False),
            )
            obj.full_weights = np.asarray(d["weights"], dtype=np.float32)
            return obj
        obj = cls(
            camera_count=d["camera_count"],
            features_per_camera=d["features_per_camera"],
            cubic=d.get("cubic", False),
            linear=d.get("linear", False),
        )
        if "full_weights" in d and d["full_weights"] is not None:
            obj.full_weights = np.asarray(d["full_weights"], dtype=np.float32)
        pcw = d.get("per_camera_weights", [])
        obj.per_camera_weights = [
            np.asarray(w, dtype=np.float32) if w is not None else None for w in pcw
        ]
        return obj


def _predict_dispatch(model, feature, active_mask=None, cubic=False, linear=False):
    """Unified prediction dispatch: handles both legacy ndarray and MultiCameraRidgeModel."""
    if isinstance(model, MultiCameraRidgeModel):
        return model.predict(feature, active_mask)
    # Legacy: raw weights ndarray
    if active_mask is not None and not all(active_mask):
        return predict(model, feature, cubic=cubic, linear=linear), "full_zero_fill"
    return predict(model, feature, cubic=cubic, linear=linear), "full"


def calibration_points(width, height):
    xs = [0.12, 0.5, 0.88]
    ys = [0.16, 0.5, 0.84]
    return [(int(width * x), int(height * y)) for y in ys for x in xs]


def make_canvas(width, height, point=None, message="", progress=0.0):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 28)
    if point is not None:
        cv2.circle(canvas, point, 18, (255, 255, 255), -1)
        cv2.circle(canvas, point, 6, (0, 0, 0), -1)
    if message:
        cv2.putText(
            canvas,
            message,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
    if progress > 0:
        bar_w = int((width - 80) * min(progress, 1.0))
        cv2.rectangle(canvas, (40, height - 52), (width - 40, height - 34), (70, 70, 70), -1)
        cv2.rectangle(canvas, (40, height - 52), (40 + bar_w, height - 34), (60, 180, 255), -1)
    return canvas


def embed_preview_strip(canvas, frames):
    if not frames:
        return canvas
    preview_h = min(160, max(90, canvas.shape[0] // 5))
    preview_w = int(preview_h * 16 / 9)
    previews = []
    for frame in frames:
        previews.append(cv2.resize(frame, (preview_w, preview_h)))
    strip = np.hstack(previews)
    max_w = canvas.shape[1] - 20
    if strip.shape[1] > max_w:
        scale = max_w / strip.shape[1]
        strip = cv2.resize(strip, (max_w, max(1, int(strip.shape[0] * scale))))
    y0 = canvas.shape[0] - strip.shape[0] - 12
    x0 = 10
    canvas[y0 - 4 : y0 + strip.shape[0] + 4, x0 - 4 : x0 + strip.shape[1] + 4] = (0, 0, 0)
    canvas[y0 : y0 + strip.shape[0], x0 : x0 + strip.shape[1]] = strip
    return canvas


def safe_named_window(name, flags=cv2.WINDOW_NORMAL):
    global HIGHGUI_AVAILABLE
    if not HIGHGUI_AVAILABLE:
        return False
    try:
        cv2.namedWindow(name, flags)
        return True
    except cv2.error as exc:
        HIGHGUI_AVAILABLE = False
        print(f"OpenCV window support is unavailable: {exc}")
        return False


def safe_set_window_property(name, prop, value):
    if not HIGHGUI_AVAILABLE:
        return False
    try:
        cv2.setWindowProperty(name, prop, value)
        return True
    except cv2.error:
        return False


def safe_imshow(name, frame):
    global HIGHGUI_AVAILABLE
    if not HIGHGUI_AVAILABLE:
        return False
    try:
        cv2.imshow(name, frame)
        return True
    except cv2.error as exc:
        HIGHGUI_AVAILABLE = False
        print(f"OpenCV cannot display windows: {exc}")
        return False


def safe_wait_key(delay=1):
    if not HIGHGUI_AVAILABLE:
        time.sleep(max(delay, 1) / 1000.0)
        return -1
    try:
        return cv2.waitKey(delay)
    except cv2.error:
        return -1


def safe_destroy_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def read_camera_features(cameras, extractors, stabilizers, mirror):
    frames = []
    feats = []
    for camera, extractor, stabilizer in zip(cameras, extractors, stabilizers):
        ok, frame, reused = camera.read()
        if not ok:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            frames.append(draw_label(blank, f"camera {camera.index}: read failed"))
            return None, frames
        feat, preview = extractor.process_frame(frame, mirror=mirror)
        if feat is None:
            if reused:
                preview = draw_label(preview, f"camera {camera.index}: reused, no eyes")
            else:
                preview = draw_label(preview, f"camera {camera.index}: eyes not detected")
            frames.append(preview)
            return None, frames
        feats.append(stabilizer.update(feat))
        if reused:
            preview = draw_label(preview, f"camera {camera.index}: detected (reused)")
        else:
            preview = draw_label(preview, f"camera {camera.index}: detected")
        frames.append(preview)
    return np.concatenate(feats), frames


def read_camera_features_partial(cameras, extractors, stabilizers, mirror):
    """Like read_camera_features() but never returns None for features.

    Cameras that fail detection get zero-filled feature slots and their
    active_mask entry is set to False. This enables partial fusion:
    gaze prediction uses only the cameras that successfully detected eyes.
    """
    frames = []
    all_feats = []
    active_mask = []
    for camera, extractor, stabilizer in zip(cameras, extractors, stabilizers):
        ok, frame, reused = camera.read()
        if not ok:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            frames.append(draw_label(blank, f"camera {camera.index}: read failed"))
            all_feats.append(np.zeros(extractor.FEATURE_COUNT, dtype=np.float32))
            active_mask.append(False)
            continue
        feat, preview = extractor.process_frame(frame, mirror=mirror)
        if feat is None:
            if reused:
                preview = draw_label(preview, f"camera {camera.index}: reused, no eyes")
            else:
                preview = draw_label(preview, f"camera {camera.index}: eyes not detected")
            frames.append(preview)
            all_feats.append(np.zeros(extractor.FEATURE_COUNT, dtype=np.float32))
            active_mask.append(False)
            continue
        all_feats.append(stabilizer.update(feat))
        if reused:
            preview = draw_label(preview, f"camera {camera.index}: detected (reused)")
        else:
            preview = draw_label(preview, f"camera {camera.index}: detected")
        frames.append(preview)
        active_mask.append(True)
    return np.concatenate(all_feats), frames, active_mask


def _threaded_read_one(camera):
    """Read a single frame from cap directly. Called in worker threads."""
    ok, frame = camera.cap.read()
    return camera, ok, frame


def read_all_camera_frames_threaded(cameras, timeout=0.05):
    """Read all camera frames in parallel threads.

    Only parallelizes cap.read(); brightness normalization and hold-last-frame
    logic run on the main thread afterward. Feature extraction stays sequential
    because MediaPipe/Haar extractors are not thread-safe.

    Returns list of (ok, frame, reused) tuples in original camera order.
    """
    import concurrent.futures

    n = len(cameras)
    if n <= 1:
        results = []
        for cam in cameras:
            ok, frame, reused = cam.read()
            results.append((ok, frame, reused))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(_threaded_read_one, cam) for cam in cameras]
        raw_results = []
        for f in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                raw_results.append(f.result())
            except Exception:
                pass

    raw_by_camera = {}
    for cam, ok, frame in raw_results:
        raw_by_camera[cam.index] = (ok, frame)

    results = []
    for cam in cameras:
        if cam.index in raw_by_camera:
            ok, frame = raw_by_camera[cam.index]
        else:
            ok, frame = False, None
        if ok and frame is not None:
            frame = cam._normalize_brightness(frame)
            cam.last_frame = frame
            cam.reads += 1
            results.append((True, frame, False))
        else:
            cam.reads += 1
            cam.failures += 1
            if cam.hold_last_frame and cam.last_frame is not None:
                cam.reused += 1
                results.append((True, cam.last_frame.copy(), True))
            else:
                results.append((False, None, False))
    return results


def _read_frame_results(cameras, args):
    """Read frames from all cameras, optionally using threads."""
    if args.threaded_reads:
        return read_all_camera_frames_threaded(cameras)
    return [(camera.read()) for camera in cameras]


def read_monitor_frames(cameras, mirror):
    frames = []
    for camera in cameras:
        ok, frame, reused = camera.read()
        if ok:
            if mirror:
                frame = cv2.flip(frame, 1)
            suffix = "reused" if reused else "monitor"
            frame = draw_label(frame, f"camera {camera.index}: {suffix}")
        else:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame = draw_label(frame, f"camera {camera.index}: read failed")
        frames.append(frame)
    return frames


def show_camera_previews(frames):
    if not frames:
        return
    previews = []
    for frame in frames:
        small = cv2.resize(frame, (320, 180))
        previews.append(small)
    safe_imshow("camera previews", np.hstack(previews))


def run_preview_only(cameras, mirror, diagnose=False):
    print("Preview mode: press q or Esc to quit.")
    last_report = time.time()
    while True:
        previews = []
        for camera in cameras:
            ok, frame, reused = camera.read()
            if ok:
                if mirror:
                    frame = cv2.flip(frame, 1)
                suffix = "reused" if reused else "OK"
                frame = draw_label(frame, f"camera {camera.index}: {suffix}")
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frame = draw_label(frame, f"camera {camera.index}: read failed")
            previews.append(cv2.resize(frame, (320, 180)))
        if diagnose and time.time() - last_report >= 2.0:
            for camera in cameras:
                stats = camera.stats()
                print(
                    f"camera {stats['index']}: fps={stats['fps']:.1f}, "
                    f"reads={stats['reads']}, failures={stats['failures']}, reused={stats['reused']}"
                )
            last_report = time.time()
        if not safe_imshow("camera previews", np.hstack(previews)):
            print("Cannot open preview window. Install GUI-enabled OpenCV, then retry.")
            break
        if safe_wait_key(1) in (27, ord("q")):
            break


def save_camera_snapshots(cameras, mirror):
    saved = 0
    for camera in cameras:
        frame = None
        ok = False
        for _ in range(20):
            ok, frame, _ = camera.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok or frame is None:
            print(f"camera {camera.index}: failed to read frame")
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        path = Path(f"camera_{camera.index}_snapshot.jpg")
        if cv2.imwrite(str(path), frame):
            print(f"camera {camera.index}: saved {path.resolve()}")
            saved += 1
        else:
            print(f"camera {camera.index}: failed to save snapshot")
    if saved == 0:
        raise SystemExit("No camera frames were saved. Check camera indexes, permissions, and USB connection.")


def run_calibration(caps, monitor_cameras, extractors, stabilizers, screen_w, screen_h, args):
    points = calibration_points(screen_w, screen_h)
    samples = []
    targets = []
    sample_masks = []
    rounds = max(1, int(args.calibration_rounds))
    total_points = len(points) * rounds
    use_partial = getattr(args, 'partial_fusion', False)
    min_active = max(1, int(getattr(args, 'min_active_cameras', 1)))
    safe_named_window("gaze", cv2.WINDOW_NORMAL)
    if not args.windowed:
        safe_set_window_property("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for round_idx in range(1, rounds + 1):
        for point_idx, point in enumerate(points, start=1):
            global_idx = (round_idx - 1) * len(points) + point_idx
            start = time.time()
            while time.time() - start < args.settle_seconds:
                canvas = make_canvas(
                    screen_w,
                    screen_h,
                    point,
                    f"Calibration {global_idx}/{total_points}: look at the dot",
                )
                if use_partial:
                    _, frames, _active = read_camera_features_partial(caps, extractors, stabilizers, args.mirror)
                else:
                    _, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
                frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
                if args.embed_previews:
                    canvas = embed_preview_strip(canvas, frames)
                safe_imshow("gaze", canvas)
                if not args.embed_previews:
                    show_camera_previews(frames)
                if safe_wait_key(1) in (27, ord("q")):
                    raise KeyboardInterrupt

            collected = 0
            while collected < args.samples_per_point:
                if use_partial:
                    feature, frames, active_mask = read_camera_features_partial(caps, extractors, stabilizers, args.mirror)
                    if sum(active_mask) < min_active:
                        feature = None
                else:
                    feature, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
                frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
                progress = collected / max(args.samples_per_point, 1)
                canvas = make_canvas(
                    screen_w,
                    screen_h,
                    point,
                    f"Calibration {global_idx}/{total_points}: keep looking",
                    progress,
                )
                if args.embed_previews:
                    canvas = embed_preview_strip(canvas, frames)
                safe_imshow("gaze", canvas)
                if not args.embed_previews:
                    show_camera_previews(frames)
                if feature is not None:
                    samples.append(feature)
                    targets.append(point)
                    if use_partial:
                        sample_masks.append(list(active_mask))
                    else:
                        sample_masks.append([True] * len(caps))
                    collected += 1
                if safe_wait_key(1) in (27, ord("q")):
                    raise KeyboardInterrupt

    if len(samples) < 20:
        raise SystemExit("Not enough calibration samples. Adjust cameras and try again.")

    cubic = getattr(args, 'cubic', False)
    linear = getattr(args, 'linear_calibration', False)
    if linear and cubic:
        print("  Note: --linear-calibration overrides --cubic.")
        cubic = False
    adaptive = getattr(args, 'adaptive_ridge', False)
    robust = getattr(args, 'robust_fit', False)
    manual_lam = getattr(args, 'ridge_lambda', None)

    # Determine lambda: manual > adaptive > default 1e-2
    if manual_lam is not None:
        lam_full = lam_per_cam = float(manual_lam)
        print(f"  Using manual lambda: {lam_full}")
    elif adaptive:
        sample_count = max(len(samples), 1)
        # Full model: 4 cams × 4 features = 16 raw → 153 quadratic → ratio ~2.6
        fpc = extractors[0].FEATURE_COUNT
        n_raw = len(caps) * fpc
        n_poly = poly_feature_count(n_raw, cubic=cubic, linear=linear)
        ratio_full = sample_count / max(n_poly, 1)
        lam_full = float(np.clip((10.0 - ratio_full) / 10.0 * 0.3 + 1e-3, 1e-3, 0.5))
        # Per-camera: 4 features → 15 quadratic → ratio ~27
        n_poly_pc = poly_feature_count(fpc, cubic=cubic, linear=linear)
        ratio_pc = sample_count / max(n_poly_pc, 1)
        lam_per_cam = float(np.clip((10.0 - ratio_pc) / 10.0 * 0.3 + 1e-4, 1e-4, 0.1))
        print(f"  Adaptive lambda: full={lam_full:.4f} (ratio={ratio_full:.1f}), per-cam={lam_per_cam:.4f} (ratio={ratio_pc:.1f})")
    else:
        lam_full = lam_per_cam = 1e-2

    if use_partial:
        fpc = extractors[0].FEATURE_COUNT
        model = MultiCameraRidgeModel(len(caps), fpc, cubic=cubic, linear=linear)
        fitter = fit_ridge_huber if robust else fit_ridge

        full_pairs = [
            (sample, target)
            for sample, target, mask in zip(samples, targets, sample_masks)
            if all(mask)
        ]
        if len(full_pairs) >= 20:
            full_samples, full_targets = zip(*full_pairs)
            print(f"  Fitting full fusion model ({len(caps)} cameras, samples={len(full_samples)}, lam={lam_full})...")
            model.full_weights = fitter(
                full_samples, full_targets, lam=lam_full, cubic=cubic, linear=linear
            )
        else:
            print(f"  Skipping full fusion model: only {len(full_pairs)} all-camera samples.")

        print(f"  Fitting per-camera models (lam={lam_per_cam})...")
        model.per_camera_weights = []
        for cam_idx in range(len(caps)):
            start = cam_idx * fpc
            end = start + fpc
            cam_pairs = [
                (sample[start:end], target)
                for sample, target, mask in zip(samples, targets, sample_masks)
                if cam_idx < len(mask) and mask[cam_idx]
            ]
            if len(cam_pairs) < 20:
                print(f"    camera {cam_idx}: skipped, only {len(cam_pairs)} active samples")
                model.per_camera_weights.append(None)
                continue
            cam_samples, cam_targets = zip(*cam_pairs)
            print(f"    camera {cam_idx}: samples={len(cam_samples)}")
            w = fitter(cam_samples, cam_targets, lam=lam_per_cam, cubic=cubic, linear=linear)
            model.per_camera_weights.append(w)

        if model.full_weights is None and not model.has_per_camera_models:
            raise SystemExit("Failed to fit calibration model.")
        validate_calibration_fit(
            model, samples, targets, screen_w, screen_h,
            cubic=cubic, linear=linear, active_masks=sample_masks,
        )
        return model, len(samples)

    fitter = fit_ridge_huber if robust else fit_ridge
    weights = fitter(samples, targets, lam=lam_full, cubic=cubic, linear=linear)
    validate_calibration_fit(weights, samples, targets, screen_w, screen_h, cubic=cubic, linear=linear)
    return weights, len(samples)


def save_calibration(path, weights, screen_w, screen_h, cubic=False, linear=False):
    if isinstance(weights, MultiCameraRidgeModel):
        data = weights.to_dict()
        data["screen_width"] = screen_w
        data["screen_height"] = screen_h
        data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        data = {
            "screen_width": screen_w,
            "screen_height": screen_h,
            "weight_rows": int(weights.shape[0]),
            "weights": weights.tolist(),
            "cubic": cubic,
            "linear": linear,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = MultiCameraRidgeModel.from_dict(data)
    cubic = data.get("cubic", False)
    linear = data.get("linear", False)
    # Check if legacy (only full_weights, no per-camera) → return raw ndarray for backward compat
    if model.per_camera_weights is None or len(model.per_camera_weights) == 0:
        return model.full_weights, cubic, linear
    return model, cubic, linear


def validate_calibration(weights, camera_count, cubic=False, linear=False, features_per_camera=4):
    def _reject_collapsed(weight_matrix):
        if weight_matrix is None or weight_matrix.shape[0] < 2:
            raise SystemExit("Calibration file is empty or invalid. Recalibrate for this camera setup.")
        non_intercept_norm = float(np.linalg.norm(weight_matrix[1:]))
        if non_intercept_norm < 1e-6:
            raise SystemExit(
                "Calibration file appears collapsed: it only predicts a constant point. "
                "Recalibrate with stable eye detection before using it."
            )

    # Accept MultiCameraRidgeModel
    if isinstance(weights, MultiCameraRidgeModel):
        if weights.full_weights is not None:
            raw_feature_count = camera_count * features_per_camera
            expected_rows = poly_feature_count(raw_feature_count, cubic=cubic, linear=linear)
            if weights.full_weights.shape[0] != expected_rows:
                raise SystemExit(
                    "Calibration file does not match the selected camera count. "
                    f"Selected cameras: {camera_count}, features/cam: {features_per_camera}, "
                    f"expected weight rows: {expected_rows}, "
                    f"file has: {weights.full_weights.shape[0]}. Recalibrate for this camera setup."
                )
            _reject_collapsed(weights.full_weights)
        if not weights.has_per_camera_models and weights.full_weights is None:
            raise SystemExit("Calibration file has no usable full or per-camera model. Recalibrate.")
        expected_per_camera_rows = poly_feature_count(features_per_camera, cubic=cubic, linear=linear)
        for idx, per_camera_weights in enumerate(weights.per_camera_weights):
            if per_camera_weights is None:
                continue
            if per_camera_weights.shape[0] != expected_per_camera_rows:
                raise SystemExit(
                    "Calibration file has a per-camera model with the wrong feature count. "
                    f"Camera slot: {idx}, expected rows: {expected_per_camera_rows}, "
                    f"file has: {per_camera_weights.shape[0]}. Recalibrate."
                )
            _reject_collapsed(per_camera_weights)
        return
    raw_feature_count = camera_count * features_per_camera
    expected_rows = poly_feature_count(raw_feature_count, cubic=cubic, linear=linear)
    if weights.shape[0] != expected_rows:
        raise SystemExit(
            "Calibration file does not match the selected camera count. "
            f"Selected cameras: {camera_count}, features/cam: {features_per_camera}, "
            f"expected weight rows: {expected_rows}, "
            f"file has: {weights.shape[0]}. Recalibrate for this camera setup."
        )
    _reject_collapsed(weights)


def run_tracking(caps, monitor_cameras, extractors, stabilizers, weights, screen_w, screen_h, args):
    use_partial = getattr(args, 'partial_fusion', False)
    use_one_euro = getattr(args, 'one_euro_filter', False)
    min_active = max(1, int(getattr(args, 'min_active_cameras', 1)))
    one_euro = None
    smooth = None
    smooth_velocity = None
    alpha_min = float(np.clip(args.gaze_alpha * 0.3, 0.02, 0.3))
    alpha_max = float(np.clip(args.gaze_alpha * 2.5, 0.3, 0.9))
    velocity_threshold = float(screen_w * 0.03)

    while True:
        if use_partial:
            feature, frames, active_mask = read_camera_features_partial(caps, extractors, stabilizers, args.mirror)
            if sum(active_mask) < min_active:
                feature = None
        else:
            feature, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
            active_mask = None
        frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
        canvas = make_canvas(screen_w, screen_h, message="Tracking: q/Esc to quit")
        if feature is not None:
            point, source = _predict_dispatch(weights, feature, active_mask,
                                              cubic=getattr(args, 'cubic', False),
                                              linear=getattr(args, 'linear_calibration', False))
            if source == "none":
                feature = None
            else:
                point[0] = np.clip(point[0], 0, screen_w - 1)
                point[1] = np.clip(point[1], 0, screen_h - 1)

        if feature is not None:
            if use_one_euro:
                if one_euro is None:
                    one_euro = OneEuroFilter(
                        min_cutoff=getattr(args, 'euro_min_cutoff', 1.0),
                        beta=getattr(args, 'euro_beta', 0.007),
                    )
                smooth = one_euro.update(point, time.time())
            elif smooth is None:
                smooth = point.copy()
                smooth_velocity = 0.0
            else:
                raw_velocity = float(np.linalg.norm(point - smooth))
                smooth_velocity = (1 - 0.3) * smooth_velocity + 0.3 * raw_velocity
                ratio = np.clip(smooth_velocity / max(velocity_threshold, 1), 0, 1)
                alpha = alpha_min + (alpha_max - alpha_min) * ratio
                smooth = (1 - alpha) * smooth + alpha * point

            xy = tuple(smooth.astype(int))
            cv2.circle(canvas, xy, 16, (0, 0, 255), -1)
            cv2.circle(canvas, xy, 26, (255, 255, 255), 2)
            cv2.putText(canvas, source, (40, screen_h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                canvas,
                "Eyes not detected",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 160, 255),
                2,
                cv2.LINE_AA,
            )
        if args.embed_previews:
            canvas = embed_preview_strip(canvas, frames)
        safe_imshow("gaze", canvas)
        if not args.embed_previews:
            show_camera_previews(frames)
        if safe_wait_key(1) in (27, ord("q")):
            break


def default_validation_points(screen_w, screen_h):
    """Return a list of 5 validation points (pixel coords) offset from the calibration grid."""
    raw = [
        (0.25, 0.30),
        (0.75, 0.30),
        (0.50, 0.50),
        (0.25, 0.70),
        (0.75, 0.70),
    ]
    return [(int(screen_w * x), int(screen_h * y)) for x, y in raw]


def parse_validation_points(raw, screen_w, screen_h):
    """Parse comma-separated normalized coords into pixel positions."""
    if not raw:
        return default_validation_points(screen_w, screen_h)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) % 2 != 0:
        raise SystemExit("--validate-points must have an even number of values.")
    points = []
    for i in range(0, len(parts), 2):
        x = float(parts[i])
        y = float(parts[i + 1])
        points.append((int(screen_w * x), int(screen_h * y)))
    return points


def run_validation(caps, monitor_cameras, extractors, stabilizers, weights, screen_w, screen_h, args):
    """Run a validation phase: show known points, collect predictions, compute error."""
    validate_points = parse_validation_points(args.validate_points, screen_w, screen_h)
    samples = max(1, int(args.validate_samples))
    settle = max(0.1, float(args.validate_settle))

    safe_named_window("gaze", cv2.WINDOW_NORMAL)
    if not args.windowed:
        safe_set_window_property("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    validation_data = {
        "config": args.use_cameras,
        "camera_indexes": selected_camera_indexes(args),
        "screen_width": screen_w,
        "screen_height": screen_h,
        "calibration_file": args.calibration,
        "samples_per_point": samples,
        "settle_seconds": settle,
        "validate_points": [{"id": f"p{i}", "gt_x": px, "gt_y": py} for i, (px, py) in enumerate(validate_points)],
        "results": [],
    }

    for point_idx, (px, py) in enumerate(validate_points):
        point_id = f"p{point_idx}"
        print(f"  Validation point {point_idx + 1}/{len(validate_points)}: ({px}, {py})")

        use_partial = getattr(args, 'partial_fusion', False)
        min_active = max(1, int(getattr(args, 'min_active_cameras', 1)))

        # Settle phase
        settle_start = time.time()
        while time.time() - settle_start < settle:
            canvas = make_canvas(
                screen_w, screen_h,
                (px, py),
                f"Validation {point_idx + 1}/{len(validate_points)}: look at the dot",
            )
            if use_partial:
                _, frames, _active = read_camera_features_partial(caps, extractors, stabilizers, args.mirror)
            else:
                _, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
            frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
            if args.embed_previews:
                canvas = embed_preview_strip(canvas, frames)
            safe_imshow("gaze", canvas)
            if not args.embed_previews:
                show_camera_previews(frames)
            if safe_wait_key(1) in (27, ord("q")):
                raise KeyboardInterrupt

        # Sample collection
        predictions = []
        timestamps = []
        detection_ok = []
        active_counts = []
        prediction_sources = []
        t0 = time.time()
        while len(predictions) < samples:
            if use_partial:
                feature, frames, active_mask = read_camera_features_partial(caps, extractors, stabilizers, args.mirror)
                active_count = sum(active_mask)
                if active_count < min_active:
                    feature = None
            else:
                feature, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
                active_mask = None
                active_count = len(caps) if feature is not None else 0
            frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
            progress = len(predictions) / max(samples, 1)
            canvas = make_canvas(
                screen_w, screen_h,
                (px, py),
                f"Validation {point_idx + 1}/{len(validate_points)}: collecting ({len(predictions)}/{samples})",
                progress,
            )
            if args.embed_previews:
                canvas = embed_preview_strip(canvas, frames)
            safe_imshow("gaze", canvas)
            if not args.embed_previews:
                show_camera_previews(frames)

            if feature is not None:
                pred, source = _predict_dispatch(weights, feature, active_mask,
                                                 cubic=getattr(args, 'cubic', False),
                                                 linear=getattr(args, 'linear_calibration', False))
                if source == "none":
                    predictions.append([-1.0, -1.0])
                    detection_ok.append(False)
                else:
                    predictions.append([float(pred[0]), float(pred[1])])
                    detection_ok.append(True)
                prediction_sources.append(source)
            else:
                predictions.append([-1.0, -1.0])
                detection_ok.append(False)
                prediction_sources.append("insufficient_active")
            active_counts.append(int(active_count))
            timestamps.append(time.time() - t0)

            if safe_wait_key(1) in (27, ord("q")):
                raise KeyboardInterrupt

        validation_data["results"].append({
            "point_id": point_id,
            "gt_x": px,
            "gt_y": py,
            "predictions": predictions,
            "timestamps": timestamps,
            "detection_ok": detection_ok,
            "active_counts": active_counts,
            "prediction_sources": prediction_sources,
        })

    metrics = compute_accuracy_metrics(validation_data)
    validation_data["metrics"] = metrics

    # Print summary
    print("")
    print("=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    print(f"  RMSE:                {metrics['rmse_px']:7.2f} px  ({metrics['rmse_deg']:.2f} deg)")
    print(f"  Mean Error:          {metrics['mean_error_px']:7.2f} px  ({metrics['mean_error_deg']:.2f} deg)")
    print(f"  P95 Error:           {metrics['p95_error_px']:7.2f} px  ({metrics['p95_error_deg']:.2f} deg)")
    print(f"  Detection Rate:      {metrics['detection_rate']:7.1f} %")
    print(f"  Precision (std):     {metrics['precision_std_px']:7.2f} px")
    print(f"  Valid samples:        {metrics['valid_samples']:6d}")
    print("=" * 60)
    print("  Per-point breakdown:")
    for pm in metrics["per_point"]:
        mean_str = f"{pm['mean_error_px']:6.2f}" if pm['mean_error_px'] is not None else "   N/A"
        std_str = f"{pm['std_px']:5.2f}" if pm['std_px'] is not None else "  N/A"
        print(
            f"    {pm['point_id']}: ({pm['gt_x']:4d},{pm['gt_y']:4d})  "
            f"mean_err={mean_str}px  "
            f"std={std_str}px  "
            f"det_rate={pm['detection_rate']:5.1f}%"
        )
    print("=" * 60)

    # Save if requested
    output_path = args.validate_output
    if not output_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"validation_{args.use_cameras}_{ts}.json"
    validation_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(output_path).write_text(json.dumps(validation_data, indent=2), encoding="utf-8")
    print(f"\nValidation data saved to: {output_path}")

    safe_destroy_windows()
    return validation_data


def compute_accuracy_metrics(data):
    """Compute RMSE, mean error, P95, detection rate, and per-point stability from validation data."""
    results = data.get("results", [])
    screen_w = data.get("screen_width", 1920)
    screen_h = data.get("screen_height", 1080)
    diag = np.sqrt(screen_w ** 2 + screen_h ** 2)

    all_errors_px = []
    per_point = []
    valid_total = 0
    total_samples = 0

    for r in results:
        gt = np.array([r["gt_x"], r["gt_y"]], dtype=np.float32)
        preds = r.get("predictions", [])
        det_ok = r.get("detection_ok", [True] * len(preds))
        total_samples += len(preds)

        point_errors = []
        for pred, ok in zip(preds, det_ok):
            if ok and pred[0] >= 0 and pred[1] >= 0:
                err = float(np.linalg.norm(np.array(pred, dtype=np.float32) - gt))
                point_errors.append(err)
                all_errors_px.append(err)

        valid_count = len(point_errors)
        valid_total += valid_count

        if point_errors:
            errors_arr = np.array(point_errors)
            per_point.append({
                "point_id": r["point_id"],
                "gt_x": r["gt_x"],
                "gt_y": r["gt_y"],
                "mean_error_px": float(np.mean(errors_arr)),
                "std_px": float(np.std(errors_arr)),
                "rmse_px": float(np.sqrt(np.mean(errors_arr ** 2))),
                "p95_px": float(np.percentile(errors_arr, 95)),
                "detection_rate": 100.0 * valid_count / max(len(preds), 1),
            })
        else:
            per_point.append({
                "point_id": r["point_id"],
                "gt_x": r["gt_x"],
                "gt_y": r["gt_y"],
                "mean_error_px": None,
                "std_px": None,
                "rmse_px": None,
                "p95_px": None,
                "detection_rate": 0.0,
            })

    if all_errors_px:
        errors = np.array(all_errors_px)
        rmse_px = float(np.sqrt(np.mean(errors ** 2)))
        mean_err_px = float(np.mean(errors))
        p95_px = float(np.percentile(errors, 95))
        precision_std_px = float(np.std(errors))
    else:
        rmse_px = mean_err_px = p95_px = precision_std_px = float("nan")

    detection_rate = 100.0 * valid_total / max(total_samples, 1)

    return {
        "rmse_px": rmse_px,
        "rmse_deg": rmse_px / diag * 180.0 / np.pi * 4.0,  # rough deg estimate
        "mean_error_px": mean_err_px,
        "mean_error_deg": mean_err_px / diag * 180.0 / np.pi * 4.0,
        "p95_error_px": p95_px,
        "p95_error_deg": p95_px / diag * 180.0 / np.pi * 4.0,
        "detection_rate": detection_rate,
        "precision_std_px": precision_std_px,
        "valid_samples": valid_total,
        "total_samples": total_samples,
        "per_point": per_point,
    }


def run_latency_test(caps, monitor_cameras, extractors, stabilizers, weights, screen_w, screen_h, args):
    """Measure end-to-end latency: detect gaze response delay after a stimulus jumps."""
    trials = max(1, int(args.latency_trials))
    safe_named_window("gaze", cv2.WINDOW_NORMAL)
    if not args.windowed:
        safe_set_window_property("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"\n{'=' * 50}")
    print(f"  LATENCY TEST: {trials} trials")
    print(f"{'=' * 50}")

    results = []
    # Two fixed positions far apart for clear signal
    pos_a = (int(screen_w * 0.2), int(screen_h * 0.5))
    pos_b = (int(screen_w * 0.8), int(screen_h * 0.5))

    for trial in range(1, trials + 1):
        print(f"\n  Trial {trial}/{trials}: look at the LEFT dot, wait for it to jump RIGHT")

        # Show dot at position A, wait for user to fixate
        settle_start = time.time()
        while time.time() - settle_start < 1.5:
            canvas = make_canvas(screen_w, screen_h, pos_a,
                                 f"Latency {trial}/{trials}: look at the dot (jumps soon)")
            _, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
            frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
            if args.embed_previews:
                canvas = embed_preview_strip(canvas, frames)
            safe_imshow("gaze", canvas)
            if safe_wait_key(1) in (27, ord("q")):
                raise KeyboardInterrupt

        # Record pre-jump gaze (baseline)
        baseline_preds = []
        for _ in range(20):
            feat, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
            frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
            canvas = make_canvas(screen_w, screen_h, pos_a,
                                 f"Latency {trial}/{trials}: jumping...")
            if args.embed_previews:
                canvas = embed_preview_strip(canvas, frames)
            safe_imshow("gaze", canvas)
            if feat is not None:
                baseline_preds.append(_predict_dispatch(
                    weights, feat, None,
                    cubic=getattr(args, 'cubic', False),
                    linear=getattr(args, 'linear_calibration', False),
                )[0])
            safe_wait_key(1)

        # JUMP: switch to position B, record timestamp
        jump_time = time.time()
        trial_data = {"trial": trial, "jump_time": jump_time,
                      "pos_a": pos_a, "pos_b": pos_b,
                      "predictions": []}

        # Record post-jump gaze trajectory
        collect_until = jump_time + 3.0
        while time.time() < collect_until:
            feat, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
            frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
            elapsed = time.time() - jump_time
            canvas = make_canvas(screen_w, screen_h, pos_b,
                                 f"Latency {trial}/{trials}: follow the dot! ({elapsed:.1f}s)")
            if feat is not None:
                pred, _ = _predict_dispatch(
                    weights, feat, None,
                    cubic=getattr(args, 'cubic', False),
                    linear=getattr(args, 'linear_calibration', False),
                )
                trial_data["predictions"].append({
                    "t": elapsed,
                    "x": float(pred[0]),
                    "y": float(pred[1]),
                })
                # Draw actual gaze position too
                cv2.circle(canvas, (int(np.clip(pred[0], 0, screen_w - 1)),
                                    int(np.clip(pred[1], 0, screen_h - 1))),
                           8, (0, 0, 255), -1)
            if args.embed_previews:
                canvas = embed_preview_strip(canvas, frames)
            safe_imshow("gaze", canvas)
            if safe_wait_key(1) in (27, ord("q")):
                raise KeyboardInterrupt

        # Analyze latency from this trial
        preds = trial_data["predictions"]
        if len(preds) >= 5:
            # Find when gaze crosses halfway from A to B
            b_pos = np.array(pos_b, dtype=np.float32)
            a_pos = np.array(pos_a, dtype=np.float32)
            half_dist = np.linalg.norm(b_pos - a_pos) * 0.5
            latency = None
            for p in preds:
                dist_to_b = np.linalg.norm(np.array([p["x"], p["y"]]) - b_pos)
                if dist_to_b < half_dist:
                    latency = p["t"]
                    break

            if latency is not None:
                print(f"    Latency: {latency * 1000:.0f} ms")
                trial_data["latency_ms"] = latency * 1000
            else:
                print(f"    Latency: could not detect gaze transition")
                trial_data["latency_ms"] = None
        else:
            print(f"    Insufficient predictions for analysis")
            trial_data["latency_ms"] = None

        results.append(trial_data)
        safe_wait_key(500)  # Brief pause

    # Summary
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    print(f"\n{'=' * 50}")
    if latencies:
        print(f"  LATENCY SUMMARY")
        print(f"  Mean:   {np.mean(latencies):.0f} ms")
        print(f"  Median: {np.median(latencies):.0f} ms")
        print(f"  Std:    {np.std(latencies):.0f} ms")
        print(f"  Min:    {np.min(latencies):.0f} ms")
        print(f"  Max:    {np.max(latencies):.0f} ms")
    else:
        print("  No valid latency measurements.")
    print(f"{'=' * 50}")

    safe_destroy_windows()
    return results


def run_drift_test(caps, monitor_cameras, extractors, stabilizers, weights, screen_w, screen_h, args):
    """Measure gaze prediction drift: fixate a single point and track prediction over time."""
    duration = max(5.0, float(args.drift_duration))
    safe_named_window("gaze", cv2.WINDOW_NORMAL)
    if not args.windowed:
        safe_set_window_property("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    center = (screen_w // 2, screen_h // 2)

    print(f"\n{'=' * 50}")
    print(f"  DRIFT TEST: {duration:.0f} seconds")
    print(f"  Fixate the center dot. Keep your head still.")
    print(f"{'=' * 50}")

    # Countdown
    for i in range(3, 0, -1):
        canvas = make_canvas(screen_w, screen_h, center, f"Starting in {i}...")
        safe_imshow("gaze", canvas)
        safe_wait_key(1000)

    predictions = []
    timestamps = []
    t0 = time.time()
    last_report = t0

    while time.time() - t0 < duration:
        feat, frames = read_camera_features(caps, extractors, stabilizers, args.mirror)
        frames = (frames or []) + read_monitor_frames(monitor_cameras, args.mirror)
        elapsed = time.time() - t0

        if feat is not None:
            pred, _ = _predict_dispatch(
                weights, feat, None,
                cubic=getattr(args, 'cubic', False),
                linear=getattr(args, 'linear_calibration', False),
            )
            predictions.append([float(pred[0]), float(pred[1])])
            timestamps.append(elapsed)

        # Show progress and current gaze
        canvas = make_canvas(screen_w, screen_h, center,
                             f"Drift test: {elapsed:.0f}s / {duration:.0f}s  (keep looking at dot)")
        if predictions:
            px, py = int(np.clip(predictions[-1][0], 0, screen_w - 1)), \
                     int(np.clip(predictions[-1][1], 0, screen_h - 1))
            cv2.circle(canvas, (px, py), 8, (0, 0, 255), -1)
            # Draw trail of recent predictions
            trail = predictions[-150:]
            for i in range(1, len(trail)):
                p1 = (int(np.clip(trail[i - 1][0], 0, screen_w - 1)),
                      int(np.clip(trail[i - 1][1], 0, screen_h - 1)))
                p2 = (int(np.clip(trail[i][0], 0, screen_w - 1)),
                      int(np.clip(trail[i][1], 0, screen_h - 1)))
                cv2.line(canvas, p1, p2, (0, 100, 255), 1)

        if args.embed_previews:
            canvas = embed_preview_strip(canvas, frames)
        safe_imshow("gaze", canvas)
        if not args.embed_previews:
            show_camera_previews(frames)

        # Periodic status update
        if elapsed - last_report >= 5.0 and predictions:
            recent = np.array(predictions[-100:]) if len(predictions) >= 100 else np.array(predictions)
            dist_to_center = np.mean(np.linalg.norm(recent - np.array(center), axis=1))
            print(f"  [{elapsed:.0f}s] samples={len(predictions)}, "
                  f"mean dist to center={dist_to_center:.1f}px")
            last_report = elapsed

        if safe_wait_key(1) in (27, ord("q")):
            raise KeyboardInterrupt

    if len(predictions) < 10:
        print("  Not enough predictions for drift analysis.")
        safe_destroy_windows()
        return None

    pred_arr = np.array(predictions)
    ts_arr = np.array(timestamps)

    # Linear drift analysis
    drift_x, intercept_x = np.polyfit(ts_arr, pred_arr[:, 0], 1)
    drift_y, intercept_y = np.polyfit(ts_arr, pred_arr[:, 1], 1)
    drift_rate = np.sqrt(drift_x ** 2 + drift_y ** 2)  # px/s

    # Total drift over the test period
    x_start = intercept_x
    x_end = intercept_x + drift_x * duration
    y_start = intercept_y
    y_end = intercept_y + drift_y * duration
    total_drift = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

    # RMS variation around the linear trend
    trend_x = intercept_x + drift_x * ts_arr
    trend_y = intercept_y + drift_y * ts_arr
    residuals = np.sqrt((pred_arr[:, 0] - trend_x) ** 2 + (pred_arr[:, 1] - trend_y) ** 2)
    rms_jitter = float(np.sqrt(np.mean(residuals ** 2)))

    # Print results
    print(f"\n{'=' * 50}")
    print(f"  DRIFT TEST RESULTS ({len(predictions)} samples over {duration:.0f}s)")
    print(f"{'=' * 50}")
    print(f"  Drift rate:          {drift_rate:.2f} px/s")
    print(f"  Total drift:         {total_drift:.1f} px  ({total_drift / 34.7:.2f} deg)")
    print(f"  RMS jitter:          {rms_jitter:.1f} px")
    print(f"  Mean distance to center: {np.mean(np.linalg.norm(pred_arr - np.array(center), axis=1)):.1f} px")
    print(f"  STD distance to center: {np.std(np.linalg.norm(pred_arr - np.array(center), axis=1)):.1f} px")
    print(f"{'=' * 50}")

    safe_destroy_windows()
    return {
        "duration_s": duration,
        "samples": len(predictions),
        "drift_rate_px_per_s": float(drift_rate),
        "total_drift_px": float(total_drift),
        "rms_jitter_px": rms_jitter,
        "mean_dist_to_center_px": float(np.mean(np.linalg.norm(pred_arr - np.array(center), axis=1))),
    }


def run_benchmark_configs(args):
    """Run calibration + validation for each camera configuration, collecting comparison data."""
    configs = [c.strip() for c in args.benchmark_configs.split(",") if c.strip()]
    if len(configs) < 2:
        raise SystemExit("--benchmark-configs requires at least 2 configurations to compare, e.g. 'left,all'.")

    trials = max(1, int(args.benchmark_trials))
    all_results = []
    screen_w, screen_h = get_screen_size(args)

    print("=" * 60)
    print("  BENCHMARK MODE")
    print(f"  Configs: {configs}")
    print(f"  Trials per config: {trials}")
    print("=" * 60)

    for trial in range(1, trials + 1):
        for config in configs:
            print(f"\n{'─' * 50}")
            print(f"  Trial {trial}/{trials}  |  Config: {config}")
            print(f"{'─' * 50}")

            # Override --use-cameras for this run
            args.use_cameras = config
            camera_indexes = selected_camera_indexes(args)
            require_gaze_cameras(camera_indexes, args)

            cal_file = f"calibration_{config}_trial{trial}.json"
            args.calibration = cal_file

            cameras = [
                CameraStream(
                    idx,
                    open_camera(idx, args),
                    hold_last_frame=not args.no_hold_last_frame,
                    normalize_brightness=args.normalize_brightness,
                    target_brightness=args.target_brightness,
                    brightness_alpha=args.brightness_alpha,
                )
                for idx in camera_indexes
            ]
            monitor_indexes = [
                idx for idx in parse_index_list(args.monitor_camera_indexes)
                if idx not in camera_indexes
            ]
            monitor_cameras = [
                CameraStream(
                    idx,
                    open_camera(idx, args),
                    hold_last_frame=not args.no_hold_last_frame,
                    normalize_brightness=args.normalize_brightness,
                    target_brightness=args.target_brightness,
                    brightness_alpha=args.brightness_alpha,
                )
                for idx in monitor_indexes
            ]

            try:
                extractors = create_extractors(
                    args.backend, len(cameras),
                    haar_face_min_neighbors=args.haar_face_min_neighbors,
                    haar_eye_min_neighbors=args.haar_eye_min_neighbors,
                    haar_face_scale=args.haar_face_scale,
                    haar_eye_scale=args.haar_eye_scale,
                    haar_face_min_size=args.haar_face_min_size,
                    mp_no_head_pose=args.mp_no_head_pose,
                    adaptive_pupil=getattr(args, 'adaptive_pupil', False),
                )
                stabilizers = [
                    FeatureStabilizer(args.feature_alpha, args.max_feature_jump)
                    for _ in cameras
                ]

                # Calibration
                weights, sample_count = run_calibration(
                    cameras, monitor_cameras, extractors, stabilizers,
                    screen_w, screen_h, args,
                )
                save_calibration(
                    cal_file, weights, screen_w, screen_h,
                    cubic=getattr(args, 'cubic', False),
                    linear=getattr(args, 'linear_calibration', False),
                )
                print(f"  Calibration saved: {cal_file} ({sample_count} samples)")

                # Validation
                val_data = run_validation(
                    cameras, monitor_cameras, extractors, stabilizers, weights,
                    screen_w, screen_h, args,
                )
                val_data["config"] = config
                val_data["trial"] = trial
                all_results.append(val_data)

            except KeyboardInterrupt:
                print("\nBenchmark interrupted.")
                break
            finally:
                for c in cameras:
                    c.release()
                for c in monitor_cameras:
                    c.release()
                for e in extractors:
                    e.close()
                safe_destroy_windows()

    # Save aggregate results
    benchmark_summary = {
        "benchmark_configs": configs,
        "trials": trials,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screen_width": screen_w,
        "screen_height": screen_h,
        "results": [
            {
                "config": r["config"],
                "trial": r.get("trial", 0),
                "rmse_px": r["metrics"]["rmse_px"],
                "mean_error_px": r["metrics"]["mean_error_px"],
                "p95_error_px": r["metrics"]["p95_error_px"],
                "detection_rate": r["metrics"]["detection_rate"],
                "precision_std_px": r["metrics"]["precision_std_px"],
                "valid_samples": r["metrics"]["valid_samples"],
            }
            for r in all_results
        ],
    }

    # Per-config aggregate
    config_aggs = {}
    for r in all_results:
        cfg = r["config"]
        if cfg not in config_aggs:
            config_aggs[cfg] = []
        config_aggs[cfg].append(r["metrics"])
    benchmark_summary["config_summary"] = {}
    for cfg, metrics_list in config_aggs.items():
        vals = {
            "rmse_px": [m["rmse_px"] for m in metrics_list if not np.isnan(m["rmse_px"])],
            "mean_error_px": [m["mean_error_px"] for m in metrics_list if not np.isnan(m["mean_error_px"])],
            "p95_error_px": [m["p95_error_px"] for m in metrics_list if not np.isnan(m["p95_error_px"])],
            "precision_std_px": [m["precision_std_px"] for m in metrics_list if not np.isnan(m["precision_std_px"])],
            "detection_rate": [m["detection_rate"] for m in metrics_list if not np.isnan(m["detection_rate"])],
        }
        benchmark_summary["config_summary"][cfg] = {
            key: {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": arr}
            for key, arr in vals.items() if arr
        }

    Path(args.benchmark_output).write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark results saved to: {args.benchmark_output}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON")
    print("=" * 70)
    header = f"  {'Config':<18} {'RMSE':>8}  {'Mean Err':>8}  {'P95':>8}  {'Det Rate':>8}  {'Precision':>8}"
    print(header)
    print("  " + "-" * 64)
    for cfg in configs:
        if cfg in config_aggs:
            metrics_list = config_aggs[cfg]
            rmse_vals = [m["rmse_px"] for m in metrics_list if not np.isnan(m["rmse_px"])]
            mean_vals = [m["mean_error_px"] for m in metrics_list if not np.isnan(m["mean_error_px"])]
            p95_vals = [m["p95_error_px"] for m in metrics_list if not np.isnan(m["p95_error_px"])]
            det_vals = [m["detection_rate"] for m in metrics_list if not np.isnan(m["detection_rate"])]
            prec_vals = [m["precision_std_px"] for m in metrics_list if not np.isnan(m["precision_std_px"])]

            rmse_str = f"{np.mean(rmse_vals):.1f}+/-{np.std(rmse_vals):.1f}" if rmse_vals else "N/A"
            mean_str = f"{np.mean(mean_vals):.1f}+/-{np.std(mean_vals):.1f}" if mean_vals else "N/A"
            p95_str = f"{np.mean(p95_vals):.1f}+/-{np.std(p95_vals):.1f}" if p95_vals else "N/A"
            det_str = f"{np.mean(det_vals):.1f}%" if det_vals else "N/A"
            prec_str = f"{np.mean(prec_vals):.1f}+/-{np.std(prec_vals):.1f}" if prec_vals else "N/A"

            print(f"  {cfg:<18} {rmse_str:>8}  {mean_str:>8}  {p95_str:>8}  {det_str:>8}  {prec_str:>8}")
    print("=" * 70)

    # Statistical test if exactly 2 configs
    if len(configs) == 2 and len(config_aggs.get(configs[0], [])) >= 2 and len(config_aggs.get(configs[1], [])) >= 2:
        run_statistical_analysis(all_results, configs[0], configs[1])

    return benchmark_summary


def run_statistical_analysis(validation_results, config_a=None, config_b=None):
    """Run paired t-test and compute Cohen's d between two configurations.

    Can be called with a list of validation result dicts (from benchmark mode)
    or by specifying two config names. Also callable standalone from --analyze.
    """
    if isinstance(validation_results, list):
        # Called from benchmark or analyze
        results = validation_results
        if config_a is None or config_b is None:
            # Auto-detect from data
            configs = list(set(r.get("config", "?") for r in results))
            if len(configs) != 2:
                print(f"\nStatistical analysis requires exactly 2 configs, found: {configs}")
                return
            config_a, config_b = configs[0], configs[1]
    else:
        return

    group_a = [r["metrics"] for r in results if r.get("config") == config_a]
    group_b = [r["metrics"] for r in results if r.get("config") == config_b]

    if len(group_a) < 2 or len(group_b) < 2:
        print(f"\nNeed at least 2 trials per config for statistical analysis. A: {len(group_a)}, B: {len(group_b)}")
        return

    # Use the minimum number of trials for pairing
    n = min(len(group_a), len(group_b))
    group_a = group_a[:n]
    group_b = group_b[:n]

    print("\n" + "=" * 60)
    print("  STATISTICAL ANALYSIS")
    print(f"  {config_a} vs {config_b}")
    print("=" * 60)

    metrics_keys = ["rmse_px", "mean_error_px", "p95_error_px", "precision_std_px", "detection_rate"]

    for key in metrics_keys:
        a_vals = np.array([m[key] for m in group_a if not np.isnan(m.get(key, float("nan")))], dtype=np.float64)
        b_vals = np.array([m[key] for m in group_b if not np.isnan(m.get(key, float("nan")))], dtype=np.float64)

        if len(a_vals) < 2 or len(b_vals) < 2:
            print(f"\n  {key}: insufficient data for test")
            continue

        # Ensure equal length for pairing
        min_len = min(len(a_vals), len(b_vals))
        a_vals = a_vals[:min_len]
        b_vals = b_vals[:min_len]

        diff = a_vals - b_vals
        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))
        t_stat = mean_diff / (sd_diff / np.sqrt(min_len)) if sd_diff > 0 else 0.0

        # Cohen's d (using pooled SD)
        pooled_sd = np.sqrt((np.var(a_vals, ddof=1) + np.var(b_vals, ddof=1)) / 2.0)
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_label = "negligible"
        elif abs_d < 0.5:
            effect_label = "small"
        elif abs_d < 0.8:
            effect_label = "medium"
        else:
            effect_label = "large"

        print(f"\n  {key}:")
        print(f"    {config_a} mean: {np.mean(a_vals):.3f} +/- {np.std(a_vals, ddof=1):.3f}")
        print(f"    {config_b} mean: {np.mean(b_vals):.3f} +/- {np.std(b_vals, ddof=1):.3f}")
        print(f"    Mean difference: {mean_diff:+.3f}")
        print(f"    Paired t({min_len - 1}): {t_stat:.3f}")
        print(f"    Cohen's d: {cohens_d:+.3f} ({effect_label})")

    print("=" * 60)


def load_validation_files(file_paths):
    """Load multiple validation JSON files into a single list."""
    results = []
    for fp in file_paths:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        # Normalize: if metrics not embedded, compute them
        if "metrics" not in data:
            data["metrics"] = compute_accuracy_metrics(data)
        results.append(data)
    return results


def main():
    args = parse_args()
    if args.list_cameras:
        found = list_cameras()
        print("Available camera indexes:", found)
        return

    # --analyze mode: offline statistical analysis from saved files
    if args.analyze:
        if not args.analyze_files:
            raise SystemExit("--analyze requires --analyze-files with comma-separated JSON paths.")
        file_paths = [p.strip() for p in args.analyze_files.split(",") if p.strip()]
        if len(file_paths) < 2:
            raise SystemExit("--analyze requires at least 2 files to compare.")
        results = load_validation_files(file_paths)
        # Print per-file metrics
        for r in results:
            m = r.get("metrics", {})
            cfg = r.get("config", "?")
            trial = r.get("trial", "?")
            print(f"  {cfg} trial{trial}: RMSE={m.get('rmse_px', 'N/A'):.1f}px  "
                  f"mean_err={m.get('mean_error_px', 'N/A'):.1f}px  "
                  f"det_rate={m.get('detection_rate', 'N/A'):.1f}%")
        run_statistical_analysis(results)
        return

    screen_w, screen_h = get_screen_size(args)
    camera_indexes = selected_camera_indexes(args)

    # --benchmark mode: full comparison across configs
    if args.benchmark:
        run_benchmark_configs(args)
        return

    require_gaze_cameras(camera_indexes, args)
    monitor_indexes = [
        index for index in parse_index_list(args.monitor_camera_indexes)
        if index not in camera_indexes
    ]
    cameras = [
        CameraStream(
            index,
            open_camera(index, args),
            hold_last_frame=not args.no_hold_last_frame,
            normalize_brightness=args.normalize_brightness,
            target_brightness=args.target_brightness,
            brightness_alpha=args.brightness_alpha,
        )
        for index in camera_indexes
    ]
    monitor_cameras = [
        CameraStream(
            index,
            open_camera(index, args),
            hold_last_frame=not args.no_hold_last_frame,
            normalize_brightness=args.normalize_brightness,
            target_brightness=args.target_brightness,
            brightness_alpha=args.brightness_alpha,
        )
        for index in monitor_indexes
    ]
    extractors = []
    stabilizers = []
    print(f"Using camera indexes: {camera_indexes}")
    if monitor_indexes:
        print(f"Monitoring extra camera indexes: {monitor_indexes}")

    try:
        if args.save_snapshots:
            save_camera_snapshots(cameras, args.mirror)
            return
        if args.preview_only or args.diagnose_cameras:
            run_preview_only(cameras, args.mirror, diagnose=args.diagnose_cameras)
            return

        extractors = create_extractors(args.backend, len(cameras),
                                       mp_no_head_pose=args.mp_no_head_pose,
                                       adaptive_pupil=getattr(args, 'adaptive_pupil', False))
        stabilizers = [
            FeatureStabilizer(args.feature_alpha, args.max_feature_jump) for _ in cameras
        ]
        if args.load_calibration:
            weights, cal_cubic, cal_linear = load_calibration(args.load_calibration)
            if cal_cubic != getattr(args, 'cubic', False):
                print(f"  Note: calibration cubic={cal_cubic}, requested cubic={args.cubic}")
            if cal_linear != getattr(args, 'linear_calibration', False):
                print(f"  Note: calibration linear={cal_linear}, requested linear={args.linear_calibration}")
            args.cubic = cal_cubic  # Use calibration's cubic setting
            args.linear_calibration = cal_linear
            validate_calibration(weights, len(cameras), cubic=args.cubic,
                                 linear=args.linear_calibration,
                                 features_per_camera=extractors[0].FEATURE_COUNT)
            print(
                f"Loaded calibration: {args.load_calibration} "
                f"(cubic={args.cubic}, linear={args.linear_calibration})"
            )
        else:
            weights, sample_count = run_calibration(
                cameras,
                monitor_cameras,
                extractors,
                stabilizers,
                screen_w,
                screen_h,
                args,
            )
            save_calibration(
                args.calibration, weights, screen_w, screen_h,
                cubic=getattr(args, 'cubic', False),
                linear=getattr(args, 'linear_calibration', False),
            )
            print(f"Saved calibration to {args.calibration} with {sample_count} samples.")

        # --validate mode: run validation after calibration
        if args.validate:
            if args.validate_output is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                args.validate_output = f"validation_{args.use_cameras}_{ts}.json"
            run_validation(
                cameras, monitor_cameras, extractors, stabilizers, weights,
                screen_w, screen_h, args,
            )
        elif args.latency_test:
            run_latency_test(
                cameras, monitor_cameras, extractors, stabilizers, weights,
                screen_w, screen_h, args,
            )
        elif args.drift_test:
            run_drift_test(
                cameras, monitor_cameras, extractors, stabilizers, weights,
                screen_w, screen_h, args,
            )
        else:
            run_tracking(cameras, monitor_cameras, extractors, stabilizers, weights, screen_w, screen_h, args)
    except KeyboardInterrupt:
        pass
    finally:
        for camera in cameras:
            camera.release()
        for camera in monitor_cameras:
            camera.release()
        for extractor in extractors:
            extractor.close()
        safe_destroy_windows()


if __name__ == "__main__":
    main()
