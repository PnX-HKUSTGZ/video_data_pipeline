"""EXIF-aware CPU image observations; editing coordinates always refer to the full image."""

import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

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
    font_size = max(12, round(max(width, height) / 90))
    font = ImageFont.load_default(size=font_size)
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
        anchor = (points[0], max(0, points[1] - font_size - 4))
        draw.rectangle(draw.textbbox(anchor, caption, font=font), fill=(20, 20, 20))
        draw.text(anchor, caption, fill=color, font=font)
    return canvas


def draw_coordinate_grid(
    canvas: Image.Image,
    full_size: tuple[int, int],
    extent: tuple[int, int, int, int],
) -> JSONObject:
    """Draw full-image coordinate rulers; a crop never restarts either axis at zero."""
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    font_size = max(12, round(max(width, height) / 85))
    font = ImageFont.load_default(size=font_size)
    ticks: dict[str, list[float]] = {}
    for axis, start, end, full, pixels in (
        ("x", extent[0], extent[2], full_size[0], width),
        ("y", extent[1], extent[3], full_size[1], height),
    ):
        lower, upper = start / full, end / full
        target = (upper - lower) / 10
        scale = 10 ** math.floor(math.log10(target))
        step = next(v * scale for v in (1, 2, 5, 10) if v * scale >= target - 1e-12)
        values = [
            round(i * step, 8)
            for i in range(
                math.ceil((lower - 1e-12) / step), math.floor((upper + 1e-12) / step) + 1
            )
        ]
        ticks[axis] = values
        for value in values:
            coordinate = max(0, min(pixels - 1, round(value * full - start)))
            caption = f"{axis}={value:.3f}"
            caption_width = draw.textbbox((0, 0), caption, font=font)[2]
            if axis == "x":
                draw.line((coordinate, 0, coordinate, height - 1), fill=(255, 255, 255, 65))
                if coordinate + caption_width + 2 > width:
                    continue
                anchor = (coordinate + 2, 2)
            else:
                draw.line((0, coordinate, width - 1, coordinate), fill=(255, 255, 255, 65))
                if coordinate < font_size + 8:
                    continue
                anchor = (2, min(max(0, height - font_size - 3), coordinate + 2))
            draw.rectangle(draw.textbbox(anchor, caption, font=font), fill=(0, 0, 0, 210))
            draw.text(anchor, caption, font=font, fill=(255, 240, 80, 255))
    return json_object(ticks)


def observe(
    image: Image.Image,
    boxes: list[Box],
    names: dict[int, str],
    max_edge: int,
    *,
    region: list[float] | None = None,
    encoding: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 95,
    coordinate_grid: bool = False,
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
    if coordinate_grid:
        metadata["grid_ticks"] = draw_coordinate_grid(rendered, (width, height), extent)
    try:
        content = [
            text_item("Unannotated image / region:"),
            image_item(raw, max_edge, encoding=encoding, jpeg_quality=jpeg_quality),
            text_item(
                "Same view with current boxes; grid labels are FULL IMAGE normalized x/y:"
                if coordinate_grid
                else "Same view with current box IDs and classes:"
            ),
            image_item(rendered, max_edge, encoding=encoding, jpeg_quality=jpeg_quality),
        ]
        return content, metadata
    finally:
        rendered.close()
        if raw is not image:
            raw.close()
