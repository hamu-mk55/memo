import cv2
from pathlib import Path

INPUT_DIR = Path("org")
OUTPUT_DIR = Path("images")  # Directory to save cropped images

# Inspection image bounds in the 1920 x 1080 screenshots.
REFERENCE_SIZE = (1920, 1080)
CROP_BOX = (320, 67, 1280, 788)  # left, top, right, bottom
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def crop_inspection_image(image):
    height, width = image.shape[:2]
    reference_width, reference_height = REFERENCE_SIZE
    left, top, right, bottom = CROP_BOX

    scale_x = width / reference_width
    scale_y = height / reference_height
    x1 = round(left * scale_x)
    y1 = round(top * scale_y)
    x2 = round(right * scale_x)
    y2 = round(bottom * scale_y)

    return image[y1:y2, x1:x2]


def main():
    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    saved_count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        cropped = crop_inspection_image(image)
        output_path = OUTPUT_DIR / image_path.relative_to(INPUT_DIR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), cropped):
            print(f"Failed to save: {output_path}")
            continue

        saved_count += 1

    print(f"Saved {saved_count} cropped images to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
