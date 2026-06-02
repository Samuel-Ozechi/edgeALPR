import base64
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
INPUT_ROOT = Path("data/license_recognition/inputs")
OUTPUT_ROOT = Path("data/license_recognition/ocr_clean")
REPORT_DIR = Path("data/license_recognition/reports")

MODEL_NAME = "gpt-4.1"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -----------------------------
# Client
# -----------------------------
client = OpenAI(
    base_url="https://ai-gateway.isw.la/v1",
    api_key="sk-0iDc3wvNLug028z6TQt20w",
)


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_image_for_ocr(image_path: Path) -> Dict[str, Any]:
    """
    Uses vision model to decide whether an image is suitable for OCR training.
    """

    image_b64 = encode_image(image_path)

    prompt = """
You are reviewing cropped license plate images for OCR model training.

The goal is to KEEP only images that look like realistic license plate crops the OCR model will see in production.

Reject images if:
- the plate/text is heavily rotated, upside down, vertical, or strongly tilted
- the image is distorted by aggressive augmentation
- characters are unreadable or partially destroyed
- blur/noise/contrast makes characters hard to distinguish
- characters are stretched, warped, cut off, or occluded
- the image does not look like a normal license plate crop

Keep images if:
- the plate is horizontally aligned or only mildly tilted
- the characters are readable
- the crop resembles a normal production input
- mild brightness, contrast, or small perspective differences are acceptable

Return only valid JSON with this exact schema:

{
  "keep": true or false,
  "quality_score": integer from 1 to 10,
  "reason": "brief explanation",
  "issue_type": "none | rotated | unreadable | distorted | blurry | cropped | not_plate | other"
}
"""

    content = [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_path.suffix.replace('.', '')};base64,{image_b64}"
            },
        },
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    return {
        "keep": bool(result.get("keep", False)),
        "quality_score": int(result.get("quality_score", 0)),
        "reason": result.get("reason", ""),
        "issue_type": result.get("issue_type", "other"),
    }


def process_split(split: str) -> list[dict]:
    input_dir = INPUT_ROOT / split
    output_dir = OUTPUT_ROOT / split
    rejected_dir = OUTPUT_ROOT / f"{split}_rejected_review"

    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    records = []

    for image_path in tqdm(image_paths, desc=f"Processing {split}"):
        try:
            result = classify_image_for_ocr(image_path)

            relative_path = image_path.relative_to(input_dir)
            clean_path = output_dir / relative_path
            reject_path = rejected_dir / relative_path

            clean_path.parent.mkdir(parents=True, exist_ok=True)
            reject_path.parent.mkdir(parents=True, exist_ok=True)

            if result["keep"]:
                shutil.copy2(image_path, clean_path)
                action = "kept"
            else:
                shutil.copy2(image_path, reject_path)
                action = "rejected"

            records.append({
                "split": split,
                "image_name": image_path.name,
                "source_path": str(image_path),
                "output_action": action,
                "keep": result["keep"],
                "quality_score": result["quality_score"],
                "issue_type": result["issue_type"],
                "reason": result["reason"],
            })

        except Exception as e:
            records.append({
                "split": split,
                "image_name": image_path.name,
                "source_path": str(image_path),
                "output_action": "error",
                "keep": False,
                "quality_score": 0,
                "issue_type": "error",
                "reason": str(e),
            })

    return records


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_records = []

    for split in ["train", "val"]:
        all_records.extend(process_split(split))

    df = pd.DataFrame(all_records)

    csv_path = REPORT_DIR / "ocr_image_filter_report.csv"
    json_path = REPORT_DIR / "ocr_image_filter_report.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\nFiltering complete.")
    print(f"Clean dataset written to: {OUTPUT_ROOT}")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")

    print("\nSummary:")
    print(df.groupby(["split", "output_action"]).size())


if __name__ == "__main__":
    main()