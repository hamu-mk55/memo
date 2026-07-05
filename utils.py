import csv
from pathlib import Path
from typing import List

import cv2
import numpy as np


class Candidate:
    def __init__(
        self,
        box: tuple,
        center: tuple,
        area: int,
        score: float,
        mean_difference: float,
        max_difference: float,
    ):
        self.box: tuple = box
        self.center: tuple = center
        self.area: int = area
        self.score: float = score
        self.mean_difference: float = mean_difference
        self.max_difference: float = max_difference
        self.confirmed: bool = False


def _morphological_operations(
    img_bin: np.ndarray,
    kernel_size: int = 3,
    morph1_operation: str = "open",
    morph2_operation: str = "close",
    erode_iterations: int = 0,
    dilate_iterations: int = 0,
) -> np.ndarray:

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if morph1_operation == "open":
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    elif morph1_operation == "close":
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    if morph2_operation == "close":
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif morph2_operation == "open":
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    if erode_iterations > 0:
        img_bin = cv2.erode(img_bin, kernel, iterations=erode_iterations)
    if dilate_iterations > 0:
        img_bin = cv2.dilate(img_bin, kernel, iterations=dilate_iterations)

    return img_bin


def create_brightness_mask(
    gray_image: np.ndarray,
    brightness_threshold: int = 100,
    binary_inv: bool = True,
    kernel_size: int = 3,
    morpho1_operation: str = "open",
    morpho2_operation: str = "close",
    erode_iterations: int = 0,
    dilate_iterations: int = 1,
):

    if not 0 <= brightness_threshold <= 255:
        raise ValueError("brightness_threshold must be between 0 and 255.")

    _, mask = cv2.threshold(
        gray_image,
        brightness_threshold,
        255,
        cv2.THRESH_BINARY_INV if binary_inv else cv2.THRESH_BINARY,
    )

    mask = _morphological_operations(
        mask,
        kernel_size=kernel_size,
        morph1_operation=morpho1_operation,
        morph2_operation=morpho2_operation,
        erode_iterations=erode_iterations,
        dilate_iterations=dilate_iterations,
    )

    return mask


def create_adaptive_mask(
    gray_image: np.ndarray,
    diff_threshold: int = 100,
    diff_kernel: tuple[int, int] = (3, 3),
    diff_mode: str = "neg",
    kernel_size: int = 3,
    morpho1_operation: str = "close",
    morpho2_operation: str = "open",
    erode_iterations: int = 0,
    dilate_iterations: int = 0,
):

    if not 0 <= diff_threshold <= 255:
        raise ValueError("diff_threshold must be between 0 and 255.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 7))
    diff = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    # ave = cv2.GaussianBlur(gray_image, (diff_kernel[0], diff_kernel[1]), 0)

    # if diff_mode == "neg":
    #     diff = cv2.subtract(ave, gray_image)
    # elif diff_mode == "pos":
    #     diff = cv2.subtract(gray_image, ave)
    # elif diff_mode == "abs":
    #     diff = cv2.absdiff(gray_image, ave)
    # else:
    #     raise ValueError(f"Invalid diff_mode: {diff_mode}")

    _, mask = cv2.threshold(
        diff,
        diff_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    mask = _morphological_operations(
        mask,
        kernel_size=kernel_size,
        morph1_operation=morpho1_operation,
        morph2_operation=morpho2_operation,
        erode_iterations=erode_iterations,
        dilate_iterations=dilate_iterations,
    )

    return mask


def create_gradient_mask(
    gray_image: np.ndarray, morph_ksize: int = 3, gradient_ksize: int = 5
):

    smoothed = cv2.GaussianBlur(gray_image, (gradient_ksize, gradient_ksize), 0)
    gradient_x = cv2.Sobel(
        smoothed,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=gradient_ksize,
    )
    gradient_y = cv2.Sobel(
        smoothed,
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=gradient_ksize,
    )

    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    gradient_8bit = cv2.normalize(
        gradient_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    _, mask = cv2.threshold(
        gradient_8bit,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((morph_ksize, morph_ksize), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.bitwise_not(mask)
    mask = cv2.erode(mask, kernel, iterations=2)

    return mask


def extract_largest_blob(roi_rotated: np.ndarray) -> np.ndarray:
    if roi_rotated is None:
        raise ValueError("roi_rotated is None")

    if roi_rotated.ndim == 3:
        roi_gray = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_rotated

    binary = np.where(roi_gray > 0, 255, 0).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if label_count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_blob = np.zeros_like(binary, dtype=np.uint8)
    largest_blob[labels == largest_label] = 255

    return largest_blob


def extract_blobs_by_area(roi_rotated: np.ndarray, minimum_area: int) -> np.ndarray:
    if roi_rotated is None:
        raise ValueError("roi_rotated is None")
    if minimum_area < 0:
        raise ValueError("minimum_area must be 0 or greater")

    if roi_rotated.ndim == 3:
        roi_gray = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_rotated

    binary = np.where(roi_gray > 0, 255, 0).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if label_count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    kept_blob = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, label_count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            kept_blob[labels == label] = 255

    return kept_blob


def detect_outer_contour(
    binary_image: np.ndarray,
    threshold: int = 127,
) -> dict | None:

    if binary_image is None:
        raise ValueError("binary_image is None")
    if binary_image.ndim == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    elif binary_image.ndim == 2:
        gray = binary_image
    else:
        raise ValueError("binary_image must be 2D or 3D image")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255.")

    h, _ = gray.shape[:2]

    left_edges = []
    right_edges = []
    widths = []

    for y in range(1, h - 1):
        profile = np.median(gray[y - 1 : y + 1, :], axis=0).astype(np.uint8)

        white_xs = np.flatnonzero(profile > threshold)
        if white_xs.size == 0:
            left_edges.append((None, y))
            right_edges.append((None, y))
            widths.append(0)
        else:
            left = int(white_xs[0])
            right = int(white_xs[-1])
            width = right - left + 1

            left_edges.append((left, y))
            right_edges.append((right, y))
            widths.append(width)

    if not widths:
        return None

    return {
        "left_edges": left_edges,
        "right_edges": right_edges,
        "widths": widths,
    }


def center_distance(first: Candidate, second: Candidate):
    return float(
        np.hypot(
            first.center[0] - second.center[0],
            first.center[1] - second.center[1],
        )
    )


def confirm_tracks(
    frame_candidates: List[List[Candidate]],
    minimum_length: int,
    maximum_gap: int,
    maximum_distance: float,
):
    tracks = []
    active_tracks = []

    for frame_index, candidates in enumerate(frame_candidates):
        unmatched = set(range(len(candidates)))

        for track in list(active_tracks):
            # Check if the track is still valid
            last_frame, last_candidate = track["detections"][-1]
            if frame_index - last_frame > maximum_gap + 1:
                active_tracks.remove(track)
                continue

            # Find  matching candidate for the track
            possible_matches = [
                candidate_index
                for candidate_index in unmatched
                if center_distance(last_candidate, candidates[candidate_index])
                <= maximum_distance
            ]
            if not possible_matches:
                continue

            # Select the best match based on distance
            match_distances = []
            for candidate_index in possible_matches:
                candidate = candidates[candidate_index]
                distance = center_distance(last_candidate, candidate)
                match_distances.append((distance, candidate_index))

            _, best_index = min(match_distances)

            # Update the track
            track["detections"].append((frame_index, candidates[best_index]))
            unmatched.remove(best_index)

        for candidate_index in unmatched:
            track = {"detections": [(frame_index, candidates[candidate_index])]}
            tracks.append(track)
            active_tracks.append(track)

    # Filter confirmed tracks
    confirmed_tracks = [
        track for track in tracks if len(track["detections"]) >= minimum_length
    ]
    for track in confirmed_tracks:
        for _, candidate in track["detections"]:
            candidate.confirmed = True

    return frame_candidates


def annotate_image(image: np.ndarray, candidates: List[Candidate]):
    result = image.copy()
    for candidate in candidates:
        color = (0, 0, 255) if candidate.confirmed else (0, 180, 255)
        label = "NG" if candidate.confirmed else "candidate"
        left, top, right, bottom = candidate.box
        margin = 5
        cv2.rectangle(
            result,
            (max(0, left - margin), max(0, top - margin)),
            (
                min(result.shape[1] - 1, right + margin),
                min(result.shape[0] - 1, bottom + margin),
            ),
            color,
            2,
        )
        cv2.putText(
            result,
            f"{label} score={candidate.score:.1f}",
            (max(5, left - 20), max(25, top - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def rotate_image(img: np.ndarray, angle_deg: float, border_value=0):
    """画像が欠けないように回転（現状、モノクロのみ）"""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # 回転後の外接矩形を計算
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 平行移動成分を調整して中心を合わせる
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


def detect_edge_by_intensity(
    gray: np.ndarray,
    intensity_thresh: int,
    pixel_thresh: int,
    intensity_direction: str,
    search_direction: str,
) -> int | None:

    if gray is None:
        raise ValueError("image is None")
    if gray.ndim != 2:
        raise ValueError("image is not single-channel")

    h, w = gray.shape[:2]

    if search_direction == "up":
        for y in range(h):
            if intensity_direction == "rise":
                count = np.count_nonzero(gray[y, :] > intensity_thresh)
            else:
                count = np.count_nonzero(gray[y, :] < intensity_thresh)

            if count >= pixel_thresh:
                return y

    elif search_direction == "down":
        for y in range(h - 1, -1, -1):
            if intensity_direction == "rise":
                count = np.count_nonzero(gray[y, :] > intensity_thresh)
            else:
                count = np.count_nonzero(gray[y, :] < intensity_thresh)

            if count >= pixel_thresh:
                return y

    elif search_direction == "left":
        for x in range(w):
            if intensity_direction == "rise":
                count = np.count_nonzero(gray[:, x] > intensity_thresh)
            else:
                count = np.count_nonzero(gray[:, x] < intensity_thresh)

            if count >= pixel_thresh:
                return x

    elif search_direction == "right":
        for x in range(w - 1, -1, -1):
            if intensity_direction == "rise":
                count = np.count_nonzero(gray[:, x] > intensity_thresh)
            else:
                count = np.count_nonzero(gray[:, x] < intensity_thresh)

            if count >= pixel_thresh:
                return x
    else:
        raise ValueError(f"illegal input: {search_direction}")

    return None


if __name__ == "__main__":
    pass
