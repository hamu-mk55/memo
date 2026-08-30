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
        frame_id: int,
    ):
        self.box: tuple = box
        self.center: tuple = center
        self.area: int = area
        self.score: float = score
        self.mean_difference: float = mean_difference
        self.max_difference: float = max_difference
        self.confirmed: bool = False

        self.frame_id = frame_id


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

    ave = cv2.GaussianBlur(gray_image, (diff_kernel[0], diff_kernel[1]), 0)

    if diff_mode == "neg":
        diff = cv2.subtract(ave, gray_image)
    elif diff_mode == "pos":
        diff = cv2.subtract(gray_image, ave)
    elif diff_mode == "abs":
        diff = cv2.absdiff(gray_image, ave)
    else:
        raise ValueError(f"Invalid diff_mode: {diff_mode}")

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


def merge_overlapping_candidates(
    frame_candidates: List[List[Candidate]],
    margin: int = 5,
    merged_score_threshold: float = 400.0,
) -> List[List[Candidate]]:
    """各フレームの候補をマージする。重なりのある候補は1つにまとめる"""

    def boxes_overlap(first: tuple, second: tuple) -> bool:
        first_left, first_top, first_right, first_bottom = first
        second_left, second_top, second_right, second_bottom = second

        x_overlap = max(first_left - margin, second_left - margin) < min(
            first_right + margin, second_right + margin
        )
        y_overlap = max(first_top - margin, second_top - margin) < min(
            first_bottom + margin, second_bottom + margin
        )

        return x_overlap and y_overlap

    merged_frames = []
    for candidates in frame_candidates:
        remaining_candidates = set(range(len(candidates)))
        merged_candidates = []

        # 候補を１つずつ取り出し、重なりのある候補をまとめる
        while remaining_candidates:
            group_indices = {remaining_candidates.pop()}
            pending = list(group_indices)

            # 探索中リスト(pending)から１つずつ取り出し、重なりのある候補を抽出
            # 抽出後、３つのリスト・集合を更新
            while pending:
                current_index = pending.pop()
                overlapping = {
                    candidate_index
                    for candidate_index in remaining_candidates
                    if boxes_overlap(
                        candidates[current_index].box,
                        candidates[candidate_index].box,
                    )
                }
                remaining_candidates.difference_update(overlapping)
                group_indices.update(overlapping)
                pending.extend(overlapping)

            group = [candidates[index] for index in sorted(group_indices)]

            # 重なりのある候補が１つしかない場合は、スコアが閾値以上ならそのまま追加
            if len(group) == 1:
                if group[0].score >= merged_score_threshold:
                    merged_candidates.append(group[0])
                continue

            # 重なりのある候補が複数ある場合は、スコア合計が閾値以上の場合に
            # バウンディングボックスをまとめて新しい候補を作成
            score_sum = sum(candidate.score for candidate in group)
            if score_sum >= merged_score_threshold:
                left = min(candidate.box[0] for candidate in group)
                top = min(candidate.box[1] for candidate in group)
                right = max(candidate.box[2] for candidate in group)
                bottom = max(candidate.box[3] for candidate in group)

                merged_candidates.append(
                    Candidate(
                        box=(left, top, right, bottom),
                        center=((left + right) / 2.0, (top + bottom) / 2.0),
                        area=sum(candidate.area for candidate in group),
                        score=score_sum,
                        mean_difference=max(
                            candidate.mean_difference for candidate in group
                        ),
                        max_difference=max(
                            candidate.max_difference for candidate in group
                        ),
                        frame_id=group[0].frame_id,
                    )
                )

        merged_frames.append(merged_candidates)

    return merged_frames


def confirm_tracks(
    frame_candidates: List[List[Candidate]],
    minimum_length: int,
    maximum_gap: int,
    maximum_distance_x: float,
    maximum_distance_y: float,
):
    """
    各フレームで検出した欠陥候補を時系列に沿ってトラックとして結合し、
    一定の長さ以上のトラックを欠陥として確定する
    """

    # track = {"detections": [(frame_index, candidate), ...]}
    tracks = []
    active_tracks = []

    # 各フレームを順に処理し、トラックを生成
    for frame_index, candidates in enumerate(frame_candidates):
        unmatched_candidates = set(range(len(candidates)))

        # 既存のアクティブトラックを処理
        for track in list(active_tracks):
            # 古いトラックを削除
            last_frame, last_candidate = track["detections"][-1]
            if frame_index - last_frame > maximum_gap + 1:
                active_tracks.remove(track)
                continue

            # 距離をもとに現在のフレームの候補とのマッチングを判断
            possible_matches = []
            for candidate_index in unmatched_candidates:
                candidate = candidates[candidate_index]
                distance_x = abs(candidate.center[0] - last_candidate.center[0])
                distance_y = abs(candidate.center[1] - last_candidate.center[1])

                allowed_distance_x = maximum_distance_x
                frame_difference = candidate.frame_id - last_candidate.frame_id
                allowed_distance_y = maximum_distance_y * frame_difference

                if (
                    distance_x <= allowed_distance_x
                    and distance_y <= allowed_distance_y
                ):
                    possible_matches.append(candidate_index)

            if not possible_matches:
                continue

            # 条件を満たす候補が複数ある場合、最も距離が近い候補を選択
            match_distances = []
            for candidate_index in possible_matches:
                candidate = candidates[candidate_index]
                distance = center_distance(last_candidate, candidate)
                match_distances.append((distance, candidate_index))

            _, best_index = min(match_distances)

            # 既存トラックを更新
            track["detections"].append((frame_index, candidates[best_index]))
            unmatched_candidates.remove(best_index)

        # 新しいトラックを作成
        for candidate_index in unmatched_candidates:
            track = {"detections": [(frame_index, candidates[candidate_index])]}
            tracks.append(track)
            active_tracks.append(track)

    # 生成したトラックの中で、一定の長さ以上のトラックを欠陥として確定
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
        if not candidate.confirmed:
            continue

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
