"""
Vision utilities: face detection, landmarks, skin tone, and face shape.
Uses OpenCV and MediaPipe for image analysis in personal styling.
"""

from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh landmark indices for key facial regions
# See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visibility.obj
LEFT_CHEEK_CENTER = 234
RIGHT_CHEEK_CENTER = 454
# Additional landmarks for cheek region boundary
LEFT_CHEEK_LANDMARKS = [234, 93, 132, 58, 136, 150]
RIGHT_CHEEK_LANDMARKS = [454, 323, 362, 288, 367, 397]
# Face shape measurement landmarks
CHIN = 152
FOREHEAD_TOP = 10
LEFT_TEMPLE = 21
RIGHT_TEMPLE = 251
LEFT_JAW = 162
RIGHT_JAW = 389
LEFT_CHEEKBONE = 93
RIGHT_CHEEKBONE = 323


def load_image(path: Path) -> np.ndarray:
    """
    Load an image from disk into a numpy array (BGR format).

    Args:
        path: Path to the image file.

    Returns:
        Image as numpy array in BGR format (OpenCV default).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image cannot be loaded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # cv2.imread returns BGR numpy array, or None if loading fails
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def detect_face(image: np.ndarray) -> Optional[dict[str, Any]]:
    """
    Detect the face in an image using MediaPipe Face Detection.

    Returns the bounding box in normalized coordinates [0, 1].
    Uses MediaPipe's BlazeFace model optimized for speed and accuracy.

    Args:
        image: Input image in BGR format (from load_image).

    Returns:
        Dict with 'bbox' as [x_min, y_min, x_max, y_max] in normalized coords,
        or None if no face detected.
    """
    # Convert BGR to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Face Detection (BlazeFace)
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=0,  # 0 = short-range, 1 = full-range
        min_detection_confidence=0.5,
    ) as face_detection:
        results = face_detection.process(image_rgb)

    if not results.detections:
        return None

    # Take the first (most prominent) face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    # Convert from relative (0-1) to absolute coordinates
    h, w, _ = image.shape
    x_min = bbox.xmin
    y_min = bbox.ymin
    x_max = bbox.xmin + bbox.width
    y_max = bbox.ymin + bbox.height

    return {
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": detection.score[0],
    }


def detect_face_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect 468 facial landmarks using MediaPipe Face Mesh.

    Landmarks cover face contour, eyes, nose, mouth, and fine-grained
    facial structure. Used for skin tone extraction and face shape estimation.

    Args:
        image: Input image in BGR format.

    Returns:
        Numpy array of shape (468, 3) with (x, y, z) per landmark in normalized
        coordinates. x, y are in [0, 1] relative to image dimensions.
        Returns None if no face detected.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    # Convert to numpy array: (468, 3) for x, y, z
    landmarks = results.multi_face_landmarks[0]
    landmarks_np = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32,
    )
    return landmarks_np


def extract_skin_tone(image: np.ndarray) -> dict[str, Any]:
    """
    Extract skin tone from cheek regions: average RGB and undertone estimate.

    Uses facial landmarks to locate left and right cheek areas, samples
    pixels from those regions, and computes average color. Undertone is
    estimated from R/G/B ratios (warm = yellow/red, cool = blue/pink,
    neutral = balanced).

    Args:
        image: Input image in BGR format.

    Returns:
        Dict with:
        - dominant_color: (R, G, B) tuple
        - undertone: 'warm' | 'cool' | 'neutral'
    """
    # Need landmarks to find cheek regions
    landmarks = detect_face_landmarks(image)
    if landmarks is None:
        return {
            "dominant_color": (0, 0, 0),
            "undertone": "neutral",
        }

    h, w, _ = image.shape

    # Get pixel coordinates of cheek centers (MediaPipe uses normalized 0-1)
    left_cheek = landmarks[LEFT_CHEEK_CENTER]
    right_cheek = landmarks[RIGHT_CHEEK_CENTER]

    # Convert normalized coords to pixel coords
    left_x = int(left_cheek[0] * w)
    left_y = int(left_cheek[1] * h)
    right_x = int(right_cheek[0] * w)
    right_y = int(right_cheek[1] * h)

    # Define cheek patch size (scale with face; use ~5% of image width)
    patch_size = max(15, int(0.05 * w))
    half = patch_size // 2

    # Extract rectangular regions around each cheek (avoid out-of-bounds)
    def safe_crop(img: np.ndarray, cx: int, cy: int, half_size: int) -> np.ndarray:
        y1 = max(0, cy - half_size)
        y2 = min(img.shape[0], cy + half_size)
        x1 = max(0, cx - half_size)
        x2 = min(img.shape[1], cx + half_size)
        return img[y1:y2, x1:x2]

    left_patch = safe_crop(image, left_x, left_y, half)
    right_patch = safe_crop(image, right_x, right_y, half)

    if left_patch.size == 0 or right_patch.size == 0:
        return {"dominant_color": (0, 0, 0), "undertone": "neutral"}

    # Combine both cheeks and compute mean BGR (OpenCV order)
    patches = np.vstack([left_patch.reshape(-1, 3), right_patch.reshape(-1, 3)])
    mean_bgr = np.mean(patches, axis=0).astype(np.uint8)

    # Convert BGR to RGB for output
    dominant_color = (
        int(mean_bgr[2]),
        int(mean_bgr[1]),
        int(mean_bgr[0]),
    )

    # Undertone estimation: use R and B channels (warm = more red, cool = more blue)
    r, g, b = dominant_color
    threshold = 15

    if r > b + threshold:
        undertone = "warm"
    elif b > r + threshold:
        undertone = "cool"
    else:
        undertone = "neutral"

    return {
        "dominant_color": dominant_color,
        "undertone": undertone,
    }


def estimate_face_shape(landmarks: np.ndarray) -> str:
    """
    Classify face shape using landmark-based ratios.

    Uses ratios of face length, width, forehead, cheekbones, and jaw
    to distinguish: Oval, Round, Square, Heart, Long.

    Args:
        landmarks: Array of shape (468, 3) from detect_face_landmarks.

    Returns:
        One of: 'oval', 'round', 'square', 'heart', 'long'
    """
    # Convert normalized coords to a consistent scale for ratio calculations
    # (normalized 0-1 is fine; we only need ratios)

    # Face length: chin to forehead
    chin_y = landmarks[CHIN][1]
    forehead_y = landmarks[FOREHEAD_TOP][1]
    face_length = abs(forehead_y - chin_y)

    # Face width: temple to temple (widest horizontal span)
    face_width = abs(landmarks[RIGHT_TEMPLE][0] - landmarks[LEFT_TEMPLE][0])

    # Forehead width (similar to face width at top)
    forehead_width = face_width

    # Cheekbone width
    cheek_width = abs(
        landmarks[RIGHT_CHEEKBONE][0] - landmarks[LEFT_CHEEKBONE][0]
    )

    # Jaw width
    jaw_width = abs(landmarks[RIGHT_JAW][0] - landmarks[LEFT_JAW][0])

    # Length-to-width ratio (how elongated the face is)
    length_width_ratio = face_length / (face_width + 1e-6)

    # Width ratios (forehead : cheekbones : jaw)
    # Heart: forehead > cheekbones > jaw
    # Oval: roughly balanced
    # Square: forehead ≈ jaw, angular
    # Round: all similar, length ≈ width

    # Classify based on ratios
    if length_width_ratio > 1.5:
        return "long"
    elif length_width_ratio < 1.1 and abs(forehead_width - jaw_width) < 0.05:
        return "round"
    elif jaw_width >= forehead_width * 0.95 and length_width_ratio < 1.3:
        return "square"
    elif forehead_width > cheek_width and jaw_width < cheek_width * 0.9:
        return "heart"
    else:
        return "oval"


def detect_body(image: np.ndarray) -> Optional[dict[str, Any]]:
    """
    Detect body / pose in image (placeholder for future implementation).

    Args:
        image: Input image in BGR format.

    Returns:
        Dict with pose keypoints or None.
    """
    # TODO: Implement with MediaPipe Pose
    return None


def extract_features(image_path: Path) -> dict[str, Any]:
    """
    Extract combined features (face, skin tone, face shape) for recommendations.

    Args:
        image_path: Path to input image.

    Returns:
        Dict with face bbox, landmarks, skin_tone, face_shape.
    """
    image = load_image(image_path)
    face = detect_face(image)
    landmarks = detect_face_landmarks(image)

    skin_tone = extract_skin_tone(image)
    face_shape = estimate_face_shape(landmarks) if landmarks is not None else None

    return {
        "face": face,
        "landmarks": landmarks,
        "skin_tone": skin_tone,
        "face_shape": face_shape,
        "body": detect_body(image),
    }
