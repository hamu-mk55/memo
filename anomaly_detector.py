import csv
import shutil
import os
import glob
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils import (
    Candidate,
    create_brightness_mask,
    create_gradient_mask,
    merge_overlapping_candidates,
    confirm_tracks,
    annotate_image,
)


def make_temporal_averaged_frame(frames, index, radius, exclude_radius):
    start = max(0, index - radius)
    end = min(len(frames), index + radius + 1)
    reference_indices = [
        reference_index
        for reference_index in range(start, end)
        if abs(reference_index - index) >= exclude_radius
    ]
    if not reference_indices:
        raise ValueError("Not enough reference frames for temporal median.")

    return np.mean(frames[reference_indices], axis=0).astype(np.float32)


def extract_candidates(
    frame,
    frame_index,
    reference,
    inspection_mask,
    pixel_threshold,
    score_threshold,
    area_min=8,
    area_max=1000,
    width_min=2,
    height_min=3,
):
    dark_difference = np.maximum(reference - frame.astype(np.float32), 0)
    binary = ((dark_difference >= pixel_threshold) & (inspection_mask > 0)).astype(
        np.uint8
    ) * 255

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = []

    for label in range(1, count):
        left, top, width, height, area = stats[label]
        if not area_min <= area <= area_max or width < width_min or height < height_min:
            continue

        label_roi = labels[top : top + height, left : left + width]
        diff_roi = dark_difference[top : top + height, left : left + width]
        values = diff_roi[label_roi == label]

        mean_difference = float(np.mean(values))
        # score = mean_difference * float(np.sqrt(area))
        score = sum(values)
        if score < score_threshold:
            continue

        center_x, center_y = centroids[label]
        candidates.append(
            Candidate(
                box=(int(left), int(top), int(left + width), int(top + height)),
                center=(float(center_x), float(center_y)),
                area=int(area),
                score=score,
                mean_difference=mean_difference,
                max_difference=float(np.max(values)),
                frame_id=frame_index,
            )
        )

    return candidates


def main(
    input_dir: str = "images",
    output_dir: str = "output2",
    debug_dir: str = "output2_debug",
    reference_radius: int = 10,
    exclude_radius: int = 6,
    pixel_threshold: int = 9,
    score_threshold: float = 200.0,
    min_track_length: int = 3,
    max_track_gap: int = 2,
    max_track_distance_x: float = 40.0,
    max_track_distance_y: float = 40.0,
    brightness_threshold: float = 100,
    image_extension: str = ".jpg",
    reset_output: bool = True,
    debug: bool = True,
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

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == image_extension
    )
    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    # Load images
    # color image for annotation/output, gray image for processing
    gray_images = []
    readable_paths = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue
        readable_paths.append(image_path)
        gray_images.append(image)

    if not readable_paths:
        raise SystemExit(f"No readable images found in: {input_dir}")

    frame_stack = np.stack(gray_images)
    frame_candidates: List[List[Candidate]] = []

    # Process each frame
    for index, frame in enumerate(frame_stack):
        # Create inspection mask
        _brightness_mask = create_brightness_mask(
            frame,
            brightness_threshold,
            binary_inv=False,
            kernel_size=5,
            morpho1_operation="open",
            morpho2_operation="none",
            dilate_iterations=0,
            erode_iterations=1,
        )
        _gradient_mask = create_gradient_mask(frame)

        inspection_mask = cv2.bitwise_and(
            _brightness_mask,
            _gradient_mask,
        )

        if debug and False:
            cv2.imwrite(
                str(debug_dir / readable_paths[index].with_suffix(".png").name),
                inspection_mask.astype(np.uint8),
            )

        # Pickup NG candidates
        reference = make_temporal_averaged_frame(
            frame_stack,
            index,
            reference_radius,
            exclude_radius,
        )

        # frame_ave = make_temporal_averaged_frame(
        #     frame_stack,
        #     index,
        #     1,
        #     0,
        # )

        frame_candidates.append(
            extract_candidates(
                frame,
                index,
                reference,
                inspection_mask,
                pixel_threshold,
                score_threshold,
            )
        )

        if debug:
            dark_difference = np.maximum(reference - frame.astype(np.float32), 0)
            cv2.imwrite(
                str(
                    debug_dir
                    / readable_paths[index]
                    .with_name(f"{readable_paths[index].stem}_diff.png")
                    .name
                ),
                dark_difference.astype(np.uint8) * 10,
            )

    # frame candiateの中で、バウンディングボックスが重複しているものをトラックとしてまとめる
    frame_candidates = merge_overlapping_candidates(frame_candidates)

    # frame_candidatesを解析し、NGかどうか判定
    frame_candidates = confirm_tracks(
        frame_candidates,
        min_track_length,
        max_track_gap,
        max_track_distance_x,
        max_track_distance_y,
    )

    # Output results
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "filename",
                "result",
                "max_score",
                "candidate_count",
                "confirmed_count",
                "boxes",
            ]
        )

        for image_path, image, candidates in zip(
            readable_paths, gray_images, frame_candidates
        ):
            confirmed = [candidate for candidate in candidates if candidate.confirmed]
            result = "NG" if confirmed else "OK"
            maximum_score = max(
                (candidate.score for candidate in candidates),
                default=0.0,
            )
            boxes = ";".join(
                ",".join(map(str, candidate.box)) for candidate in confirmed
            )
            writer.writerow(
                [
                    image_path.name,
                    result,
                    f"{maximum_score:.3f}",
                    len(candidates),
                    len(confirmed),
                    boxes,
                ]
            )

            if candidates:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                annotated = annotate_image(image, candidates)
                cv2.imwrite(str(output_dir / image_path.name), annotated)
            else:
                cv2.imwrite(str(output_dir / image_path.name), image)

    # Output detailed results
    detail_csv_path = output_dir / "result_detail.csv"
    with detail_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_number",
                "box",
                "center_x",
                "center_y",
                "area",
                "score",
                "mean_difference",
                "max_difference",
                "confirmed",
            ]
        )

        for candidates in frame_candidates:
            for candidate in candidates:
                if not candidate.confirmed:
                    continue
                writer.writerow(
                    [
                        candidate.frame_id,
                        ",".join(map(str, candidate.box)),
                        candidate.center[0],
                        candidate.center[1],
                        candidate.area,
                        candidate.score,
                        candidate.mean_difference,
                        candidate.max_difference,
                        candidate.confirmed,
                    ]
                )


if __name__ == "__main__":
    for image_dir in glob.glob(f"./images/*"):
        if not os.path.isdir(image_dir):
            continue

        dirname = os.path.basename(image_dir)
        out_dir = f"output_{dirname}"
        debug_dir = f"output_{dirname}_debug"

        main(input_dir=f"./images/{dirname}", output_dir=out_dir, debug_dir=debug_dir)
