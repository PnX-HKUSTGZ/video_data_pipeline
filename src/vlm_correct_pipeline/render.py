"""EXIF-aware CPU image observations; editing coordinates always refer to the full image."""

import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageOps

from .models import Box, JSONObject, json_object, valid_bbox


def load_image(path: Path) -> Image.Image:
    """Decode one image and detach an oriented RGB copy from its file handle."""
    with Image.open(path) as source:
        source.load()
        oriented = ImageOps.exif_transpose(source)
        try:
            return oriented.convert("RGB")
        finally:
            oriented.close()


def image_item(
    image: Image.Image,
    max_edge: int,
    *,
    encoding: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 95,
) -> JSONObject:
    """Encode a bounded observation in memory; never create disk thumbnails."""
    copy = image.copy()
    try:
        copy.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        if encoding == "jpeg":
            copy.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=0)
        else:
            copy.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:image/{encoding};base64,{encoded}",
            "detail": "high",
        }
    finally:
        copy.close()


def text_item(text: str) -> JSONObject:
    return {"type": "input_text", "text": text}


def message(content: list[JSONObject]) -> JSONObject:
    return json_object({"role": "user", "content": content})


def overlay(image: Image.Image, boxes: list[Box], names: dict[int, str]) -> Image.Image:
    """Draw stable IDs and classes without obscuring the interior of a target."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(1, round(max(width, height) / 700))
    for box in boxes:
        x1, y1, x2, y2 = box.bbox
        points = (
            min(width - 1, round(x1 * width)),
            min(height - 1, round(y1 * height)),
            min(width - 1, round(x2 * width)),
            min(height - 1, round(y2 * height)),
        )
        color = (255, 70, 30) if box.class_id % 2 == 0 else (0, 180, 255)
        draw.rectangle(points, outline=color, width=line_width)
        caption = f"{box.box_id} | {box.class_id}: {names[box.class_id]}"
        draw.text((points[0], max(0, points[1] - 14)), caption, fill=color)
    return canvas


def observe(
    image: Image.Image,
    boxes: list[Box],
    names: dict[int, str],
    max_edge: int,
    *,
    region: list[float] | None = None,
    encoding: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 95,
) -> tuple[list[JSONObject], JSONObject]:
    """Return raw/annotated views plus the actual rounded crop in full-image coordinates."""
    rendered = overlay(image, boxes, names)
    raw = image
    width, height = image.size
    extent = (0, 0, width, height)
    if region is not None:
        x1, y1, x2, y2 = valid_bbox(region)
        extent = (
            math.floor(x1 * width),
            math.floor(y1 * height),
            math.ceil(x2 * width),
            math.ceil(y2 * height),
        )
        raw = image.crop(extent)
        cropped = rendered.crop(extent)
        rendered.close()
        rendered = cropped
    metadata = json_object(
        {
            "full_image_size": [width, height],
            "crop_pixels": list(extent),
            "crop_bbox": [
                extent[0] / width,
                extent[1] / height,
                extent[2] / width,
                extent[3] / height,
            ],
            "editing_coordinate_system": "full_image_normalized_xyxy",
        }
    )
    try:
        content = [
            text_item("Unannotated image / region:"),
            image_item(raw, max_edge, encoding=encoding, jpeg_quality=jpeg_quality),
            text_item("Same view with current box IDs and classes:"),
            image_item(rendered, max_edge, encoding=encoding, jpeg_quality=jpeg_quality),
        ]
        return content, metadata
    finally:
        rendered.close()
        if raw is not image:
            raw.close()
