import csv
import shutil
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils import (
    create_brightness_mask,
    create_gradient_mask,
    create_adaptive_mask,
    extract_largest_blob,
    rotate_image,
    detect_outer_contour,
)


def detect_angle(
    roi,
    y_step=3,
    profile_height=3,
    foreground_threshold=127,
    min_width=20,
    min_points=10,
):

    height, width = roi.shape[:2]
    center_x = width // 2
    half_profile_height = max(0, profile_height // 2)

    center_points = []

    for y in range(0, height, y_step):
        band_top = max(0, y - half_profile_height)
        band_bottom = min(height, y + half_profile_height + 1)
        profile = np.max(roi[band_top:band_bottom, :], axis=0)

        is_foreground = profile > foreground_threshold

        # if the center pixel is not foreground, skip this row
        if not is_foreground[center_x]:
            continue

        prof_right = is_foreground[center_x:]
        bg_points = np.where(~prof_right)[0]
        if len(bg_points) > 0:
            right_edge = center_x + bg_points[0]
        else:
            right_edge = center_x

        prof_left = is_foreground[: center_x + 1][::-1]
        bg_points = np.where(~prof_left)[0]
        if len(bg_points) > 0:
            left_edge = center_x - bg_points[0]
        else:
            left_edge = center_x

        x_center = (left_edge + right_edge) / 2.0
        y_center = (band_top + band_bottom - 1) / 2.0
        center_points.append((y_center, x_center))

    if len(center_points) < min_points:
        return None

    ys = np.array([point[0] for point in center_points], dtype=np.float32)
    xs = np.array([point[1] for point in center_points], dtype=np.float32)
    slope, _ = np.polyfit(ys, xs, 1)
    angle = np.degrees(np.arctan(slope))

    return float(angle)


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


def main(
    input_dir="images",
    output_dir="output",
    debug_dir="debug",
    image_extension=".jpg",
    mask_threshold=100,
    inspection_area=(260, 150, 440, 470),  # left, top, width, height
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

        fw = open(output_dir / f"{work_dir.name}.csv", "w", newline="")
        writer = csv.writer(fw)
        writer.writerow(
            [
                "image_name",
                "detected_angle",
                "min_width",
                "min_y",
                "edge_left",
                "edge_right",
                "y_base",
            ]
        )

        for image_index, image_path in enumerate(image_paths):
            if image_index > 50000:
                break

            print(image_path)

            # Load images
            # color image for annotation/output, gray image for processing
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipped unreadable image: {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Pickup ROIs
            brightness_roi = create_brightness_mask(gray, mask_threshold)

            gradient_roi = create_adaptive_mask(
                gray, diff_threshold=5, diff_kernel=(21, 3), kernel_size=0
            )
            # gradient_roi = cv2.bitwise_not(gradient_roi)

            inspection_roi = cv2.bitwise_or(
                brightness_roi,
                gradient_roi,
            )
            # kernel = np.ones((3, 3), dtype=np.uint8)
            # inspection_roi = cv2.erode(inspection_roi, kernel, iterations=1)

            if False:
                output_path = debug_dir / image_path.relative_to(input_dir).with_name(
                    f"{image_path.stem}_mask{image_path.suffix}"
                )
                output_path0 = debug_dir / image_path.relative_to(input_dir).with_name(
                    f"{image_path.stem}_org{image_path.suffix}"
                )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path0.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(
                    str(output_path),
                    inspection_roi,
                )
                cv2.imwrite(
                    str(output_path0),
                    image,
                )

            # Detect angle of ROI
            left, top, width, height = inspection_area
            x0 = max(0, left)
            y0 = max(0, top)
            x1 = min(gray.shape[1], left + width)
            y1 = min(gray.shape[0], top + height)
            roi_cropped = brightness_roi[y0:y1, x0:x1]
            angle = detect_angle(roi_cropped)

            print(f"Image: {image_index}, Detected angle: {angle:.2f} degrees")

            # Rotate the image based on the detected angle
            if angle is None:
                continue

            roi_cropped = inspection_roi[y0:y1, x0:x1]
            roi_rotated = rotate_image(roi_cropped, -angle)
            roi_rotated = extract_largest_blob(roi_rotated)

            image_cropped = image[y0:y1, x0:x1]
            image_rotated = rotate_image(image_cropped, -angle)

            if debug:
                output_path = debug_dir / image_path.relative_to(input_dir).with_name(
                    f"{image_path.stem}_bin{image_path.suffix}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(
                    str(output_path),
                    roi_rotated,
                )

            # detect the edges of the target area
            y0 = 200
            y1 = 240

            ret = detect_outer_contour(roi_rotated[y0:y1, :], threshold=127)
            widths = np.array(ret["widths"])
            indices = np.flatnonzero(widths > 0)
            y_base = int(indices[-1]) if indices.size > 0 else None

            min_index = int(np.argmin(ret["widths"]))
            min_width = int(ret["widths"][min_index])
            min_y = int(ret["left_edges"][min_index][1])

            edge_left = ret["left_edges"][min_index][0]
            edge_right = ret["right_edges"][min_index][0]

            cv2.line(
                image_rotated,
                (edge_left, y0),
                (edge_left, y1),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.line(
                image_rotated,
                (edge_right, y0),
                (edge_right, y1),
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

            writer.writerow(
                [
                    image_path.name,
                    angle,
                    min_width,
                    min_y,
                    edge_left,
                    edge_right,
                    y_base,
                ]
            )
        fw.close()


if __name__ == "__main__":
    main()
