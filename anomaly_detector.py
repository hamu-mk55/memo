import csv
import shutil
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils import (
    Candidate,
    create_brightness_mask,
    create_gradient_mask,
    confirm_tracks,
    annotate_image,
)


def temporal_reference(frames, index, radius, exclude_radius):
    start = max(0, index - radius)
    end = min(len(frames), index + radius + 1)
    reference_indices = [
        reference_index
        for reference_index in range(start, end)
        if abs(reference_index - index) > exclude_radius
    ]
    if not reference_indices:
        raise ValueError("Not enough reference frames for temporal median.")

    return np.median(frames[reference_indices], axis=0).astype(np.float32)


def extract_candidates(
    frame,
    reference,
    inspection_mask,
    pixel_threshold,
    score_threshold,
    area_min=8,
    area_max=1000,
    width_min=3,
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

        values = dark_difference[labels == label]
        mean_difference = float(np.mean(values))
        score = mean_difference * float(np.sqrt(area))
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
            )
        )

    return candidates


def main(
    input_dir: str = "images",
    output_dir: str = "output2",
    debug_dir: str = "output2_debug",
    reference_radius: int = 20,
    exclude_radius: int = 4,
    pixel_threshold: int = 9,
    score_threshold: float = 50.0,
    min_track_length: int = 3,
    max_track_gap: int = 2,
    max_track_distance: float = 40.0,
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
    color_images = []
    gray_images = []
    readable_paths = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue
        readable_paths.append(image_path)
        color_images.append(image)
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

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

        if debug:
            cv2.imwrite(
                str(debug_dir / readable_paths[index].with_suffix(".png").name),
                inspection_mask.astype(np.uint8),
            )

        # Pickup NG candidates
        reference = temporal_reference(
            frame_stack,
            index,
            reference_radius,
            exclude_radius,
        )
        frame_candidates.append(
            extract_candidates(
                frame,
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

    frame_candidates = confirm_tracks(
        frame_candidates,
        min_track_length,
        max_track_gap,
        max_track_distance,
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
            readable_paths, color_images, frame_candidates
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
                annotated = annotate_image(image, candidates)
                cv2.imwrite(str(output_dir / image_path.name), annotated)
            else:
                cv2.imwrite(str(output_dir / image_path.name), image)


if __name__ == "__main__":
    main()
