import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from utils import (
    create_brightness_mask,
    create_gradient_mask,
    create_adaptive_mask,
    extract_largest_blob,
    delete_small_blob,
    rotate_image,
    detect_outer_contour,
)


def find_contour_edges_from_profile(profile: np.ndarray, threshold: int, target_x: int):
    """
    target_x を含む or 最も近い前景区間について、前景画素の左右端を返す。
    前景が存在しない場合は None を返す。
    """

    is_foreground = profile > threshold
    foreground_xs = np.flatnonzero(is_foreground)

    if foreground_xs.size == 0:
        return None

    # target_x が前景上ならその区間、背景上なら最も近い前景区間を選ぶ。
    nearest_x = int(foreground_xs[np.argmin(np.abs(foreground_xs - target_x))])

    # 左右端を探索
    left_edge = nearest_x
    while left_edge > 0 and is_foreground[left_edge - 1]:
        left_edge -= 1

    right_edge = nearest_x
    last_x = profile.size - 1
    while right_edge < last_x and is_foreground[right_edge + 1]:
        right_edge += 1

    return left_edge, right_edge


def detect_angle(
    roi,
    y_step: int = 3,
    foreground_threshold: int = 127,
    min_points: int = 10,
    width_sigma_outliner: float = 3.0,
):
    """
    中央縦方向のブロブの輪郭を検出し、傾き角度を計算する。
    幅が平均値から width_sigma_outliner 標準偏差より離れた点は、
    角度計算から除外する。
    """

    height, width = roi.shape[:2]
    center_x = width // 2
    center_y = height // 2
    center_points = []

    for y in range(0, height, y_step):
        band_top = max(0, y)
        band_bottom = min(height, y + y_step)
        profile_x = np.max(roi[band_top:band_bottom, :], axis=0)

        edges = find_contour_edges_from_profile(
            profile_x, foreground_threshold, center_x
        )
        if edges is None:
            continue

        left_edge, right_edge = edges
        x_center = (left_edge + right_edge) / 2.0
        y_center = (band_top + band_bottom - 1) / 2.0
        width = right_edge - left_edge + 1
        center_points.append((y_center, x_center, width))

    if len(center_points) < min_points:
        return None

    # 幅をもとに、外れ値を除去
    widths = np.array([point[2] for point in center_points], dtype=np.float32)
    width_mean = float(np.mean(widths))
    width_std = float(np.std(widths))

    if width_std > 0:
        width_deviation_limit = width_sigma_outliner * width_std
        center_points = [
            point
            for point in center_points
            if abs(point[2] - width_mean) <= width_deviation_limit
        ]

    if len(center_points) < min_points:
        return None

    # 中心点の座標から傾き角度を計算
    ys = np.array([point[0] for point in center_points], dtype=np.float32)
    xs = np.array([point[1] for point in center_points], dtype=np.float32)
    slope, intercept = np.polyfit(ys, xs, 1)

    angle = np.degrees(np.arctan(slope))
    offset_x = slope * center_y + intercept - center_x

    return float(angle), float(offset_x)


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


def detect_peaks_from_profile(
    profile: np.ndarray, closing_kernel_size: int = 3
) -> np.ndarray:
    """1次元のbinaryプロファイルから白帯の中心を検出する。
    白帯の検出前にOpenCVで1次元のクロージング処理を行う。
    戻り値はY座標が大きい順に並べる。
    """

    if closing_kernel_size < 1 or closing_kernel_size % 2 == 0:
        raise ValueError("closing_kernel_size must be a positive odd integer")

    profile = np.asarray(profile).reshape(-1)
    if profile.size == 0:
        return np.empty(0, dtype=np.int32)

    white_mask = profile != 0
    if closing_kernel_size > 1:
        kernel = np.ones((closing_kernel_size, 1), dtype=np.uint8)
        white_mask = cv2.morphologyEx(
            white_mask.astype(np.uint8).reshape(-1, 1),
            cv2.MORPH_CLOSE,
            kernel,
        ).reshape(-1).astype(bool)

    mask_changes = np.diff(np.pad(white_mask.astype(np.int8), (1, 1)))
    band_starts = np.flatnonzero(mask_changes == 1)
    band_ends = np.flatnonzero(mask_changes == -1) - 1
    centers = (band_starts + band_ends) // 2

    return centers[::-1].astype(np.int32)


def main(
    input_dir="images",
    output_dir="output",
    debug_dir="debug",
    image_extension=".jpg",
    mask_threshold=100,
    inspection_area=(200, 150, 470, 380),  # left, top, width, height
    reset_output=True,
    debug=True,
):
    # Setup
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    debug_dir = Path(debug_dir)

    if output_dir.exists() and reset_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        if debug_dir.exists() and reset_output:
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Each Work
    for work_dir in sorted(input_dir.iterdir()):
        if not work_dir.is_dir():
            continue

        image_paths = sorted(
            path
            for path in work_dir.glob("*")
            if path.is_file() and path.suffix.lower() == image_extension
        )

        xt_slices = []
        xt_peak_results = []
        frame_peak_results = []
        for image_index, image_path in enumerate(image_paths):
            if image_index > 10000:
                break

            # 検出できないフレームもCSV上に残す。
            frame_peak_results.append((image_path.name, []))

            # Load images
            # color image for annotation/output, gray image for processing
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipped unreadable image: {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Pickup ROIs-------------------------------------------------
            brightness_roi = create_brightness_mask(gray, mask_threshold)

            gradient_roi = create_adaptive_mask(
                gray,
                diff_threshold=10,
                diff_kernel=(15, 9),
                kernel_size=3,
                morpho1_operation="open",
                morpho2_operation="close",
            )

            inspection_roi = cv2.bitwise_or(
                brightness_roi,
                gradient_roi,
            )

            # Detect angle of ROI by brightness_roi------------------------
            left, top, width, height = inspection_area
            x0 = max(0, left)
            y0 = max(0, top)
            x1 = min(gray.shape[1], left + width)
            y1 = min(gray.shape[0], top + height)
            roi_cropped = brightness_roi[y0:y1, x0:x1]
            roi_cropped = extract_largest_blob(roi_cropped)
            detection = detect_angle(roi_cropped)

            # print(f"Image: {image_index}, Detected angle: {angle:.2f} degrees")

            if False:
                output_path = debug_dir / image_path.relative_to(input_dir).with_name(
                    f"{image_path.stem}_mask{image_path.suffix}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), roi_cropped)

            # Rotate the image based on the detected angle-------------------
            if detection is None:
                continue

            angle, offset_x = detection

            roi_cropped = inspection_roi[y0:y1, x0:x1]
            roi_rotated = rotate_image(
                roi_cropped, -angle, offset_x=offset_x, size_keep=True
            )
            roi_rotated = delete_small_blob(roi_rotated, min_area=100, min_width=20)

            image_cropped = image[y0:y1, x0:x1]
            image_rotated = rotate_image(
                image_cropped, -angle, offset_x=offset_x, size_keep=True
            )

            gray_cropped = gray[y0:y1, x0:x1]
            gray_rotated = rotate_image(
                gray_cropped, -angle, offset_x=offset_x, size_keep=True
            )

            if debug:
                output_path = debug_dir / image_path.relative_to(input_dir).with_name(
                    f"{image_path.stem}_bin{image_path.suffix}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), roi_rotated)

            # detect the edges of the target area
            _y0 = int(roi_rotated.shape[0] * 1 / 4)
            _y1 = int(roi_rotated.shape[0] * 3 / 4)

            ret = detect_outer_contour(roi_rotated[_y0:_y1, :], threshold=127)

            max_index = int(np.argmax(ret["widths"]))
            edge_left = ret["left_edges"][max_index][0]
            edge_right = ret["right_edges"][max_index][0]

            cv2.line(
                image_rotated,
                (edge_left, 0),
                (edge_left, image_rotated.shape[0]),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.line(
                image_rotated,
                (edge_right, 0),
                (edge_right, image_rotated.shape[0]),
                color=(255, 0, 0),
                thickness=2,
            )

            if debug:
                output_path = debug_dir / image_path.relative_to(input_dir).with_name(
                    f"{image_path.stem}_out{image_path.suffix}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(
                    str(output_path),
                    image_rotated,
                )

            # roi_roatedの右端から10pxのsliceを抽出
            col = roi_rotated[:, edge_right - 10 : edge_right + 10]
            profile = np.max(col, axis=1).astype(np.uint8)

            peaks = detect_peaks_from_profile(profile)

            frame_peak_results[-1] = (image_path.name, peaks.tolist())
            xt_peak_results.append(peaks)

            # XT slice
            col = gray_rotated[:, edge_right - 10 : edge_right + 10]
            profile = np.median(col, axis=1).astype(np.uint8)
            xt_slices.append(profile)

        # Save peak positions. peak_1 is always the candidate with the largest Y.
        max_peak_count = max(
            (len(peaks) for _, peaks in frame_peak_results),
            default=0,
        )
        with open(
            output_dir / f"{work_dir.name}.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as fw:
            writer = csv.writer(fw)
            writer.writerow(
                ["image_name"]
                + [f"peak_{index}" for index in range(1, max_peak_count + 1)]
            )
            for image_name, peaks in frame_peak_results:
                padding = [""] * (max_peak_count - len(peaks))
                writer.writerow([image_name] + peaks + padding)

        # Save XT image
        if len(xt_slices) > 0:
            xt_image = np.stack(xt_slices, axis=0)
            xt_image = np.transpose(xt_image, (1, 0))
            xt_image = cv2.cvtColor(xt_image, cv2.COLOR_GRAY2BGR)

            for frame_index, peaks in enumerate(xt_peak_results):
                for peak_y in peaks:
                    cv2.circle(
                        xt_image,
                        (frame_index, int(peak_y)),
                        radius=2,
                        color=(0, 0, 255),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

            output_path = output_dir / f"{work_dir.name}_xt.jpg"
            cv2.imwrite(str(output_path), xt_image)


if __name__ == "__main__":
    main()
