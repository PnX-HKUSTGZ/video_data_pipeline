"""EXIF-aware CPU image observations; editing coordinates always refer to the full image."""

import base64
import math
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import PoseConfig
from .models import Box, JSONObject, PoseBox, json_object, valid_bbox

POINT_COLORS = (
    (255, 75, 75),
    (80, 230, 100),
    (75, 140, 255),
    (255, 215, 70),
    (235, 80, 240),
    (40, 225, 235),
    (255, 160, 50),
    (170, 110, 255),
    (150, 240, 80),
    (255, 120, 180),
    (85, 200, 170),
    (210, 190, 255),
)


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


def overlay(
    image: Image.Image,
    boxes: Sequence[Box],
    names: dict[int, str],
    *,
    pose: PoseConfig | None = None,
    show_boxes: bool = True,
    _viewport: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Draw scoped annotations; point-only references never use their source boxes."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    viewport = _viewport or (0, 0, width, height)
    view_width, view_height = viewport[2] - viewport[0], viewport[3] - viewport[1]
    line_width = max(1, round(max(view_width, view_height) / 700))
    font_size = max(12, round(max(view_width, view_height) / 90))
    font = ImageFont.load_default(size=font_size)
    short_ids = {
        box.box_id: f"b{int(box.box_id[4:])}"
        if box.box_id.startswith("box_") and box.box_id[4:].isdigit()
        else box.box_id
        for box in boxes
    }
    # Keep custom identifiers intact and avoid aliases that collide with another object.
    if len(set(short_ids.values())) != len(short_ids):
        short_ids = {box.box_id: box.box_id for box in boxes}
    for box in boxes:
        if show_boxes:
            x1, y1, x2, y2 = box.bbox
            points = (
                min(width - 1, round(x1 * width)),
                min(height - 1, round(y1 * height)),
                min(width - 1, round(x2 * width)),
                min(height - 1, round(y2 * height)),
            )
            color = (255, 70, 30) if box.class_id % 2 == 0 else (0, 180, 255)
            draw.rectangle(points, outline=color, width=line_width)
            if (
                points[0] < viewport[2]
                and points[2] >= viewport[0]
                and points[1] < viewport[3]
                and points[3] >= viewport[1]
            ):
                caption = f"{box.box_id} | {box.class_id}: {names[box.class_id]}"
                anchor = (
                    max(viewport[0], points[0]),
                    max(viewport[1], points[1] - font_size - 4),
                )
                draw.rectangle(draw.textbbox(anchor, caption, font=font), fill=(20, 20, 20))
                draw.text(anchor, caption, fill=color, font=font)
        if pose is not None and isinstance(box, PoseBox):
            _draw_keypoints(
                draw,
                box,
                pose,
                short_ids[box.box_id],
                image.size,
                viewport,
                line_width,
                font,
                font_size,
            )
    return canvas


def _draw_keypoints(
    draw: ImageDraw.ImageDraw,
    box: PoseBox,
    pose: PoseConfig,
    display_id: str,
    image_size: tuple[int, int],
    viewport: tuple[int, int, int, int],
    line_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_size: int,
) -> None:
    """Show point identities and pending visibility without connecting missing slots."""
    width, height = image_size
    active = set(pose.class_keypoint_indices[box.class_id])
    locations = {
        index: (min(width - 1, round(point.x * width)), min(height - 1, round(point.y * height)))
        for index, point in enumerate(box.keypoints)
        if index in active and point.visibility != 0
    }
    locations = {
        index: (x, y)
        for index, (x, y) in locations.items()
        if viewport[0] <= x < viewport[2] and viewport[1] <= y < viewport[3]
    }
    for first, second in pose.skeleton:
        if (
            first in locations
            and second in locations
            and box.keypoints[first].visibility == 2
            and box.keypoints[second].visibility == 2
        ):
            draw.line((locations[first], locations[second]), fill=(190, 190, 190), width=line_width)
    radius = max(3, line_width * 2)
    if not locations:
        return
    center_x = sum(point[0] for point in locations.values()) / len(locations)
    center_y = sum(point[1] for point in locations.values()) / len(locations)
    for index, (x, y) in locations.items():
        point = box.keypoints[index]
        color = POINT_COLORS[index % len(POINT_COLORS)]
        pending = "~" if point.visibility == 1 else "?" if not point.reviewed else ""
        caption = f"{display_id}/{index}:{pose.kpt_names[index]}{pending}"
        caption_width = draw.textbbox((0, 0), caption, font=font)[2]
        # Place labels away from the point cluster, using no bounding-box geometry.
        dx, dy = x - center_x, y - center_y
        if abs(dx) >= abs(dy):
            anchor_x = x + radius + 3 if dx >= 0 else x - radius - 3 - caption_width
            anchor_y = y - font_size // 2
        else:
            anchor_x = x - caption_width // 2
            anchor_y = y + radius + 3 if dy >= 0 else y - radius - 3 - font_size
        anchor = (
            max(viewport[0], min(anchor_x, viewport[2] - caption_width)),
            max(viewport[1], min(anchor_y, viewport[3] - font_size - 2)),
        )
        draw.rectangle(draw.textbbox(anchor, caption, font=font), fill=(20, 20, 20))
        draw.text(anchor, caption, fill=color, font=font)
    # Draw markers last so neighboring captions cannot hide their exact pixel locations.
    for index, (x, y) in locations.items():
        extent = (x - radius, y - radius, x + radius, y + radius)
        if box.keypoints[index].visibility == 2:
            draw.ellipse(
                extent,
                fill=POINT_COLORS[index % len(POINT_COLORS)],
                outline=(20, 20, 20),
                width=line_width,
            )
        else:
            draw.ellipse(extent, outline=(255, 165, 0), width=line_width)


def focus_region(box: Box, pose: PoseConfig | None = None) -> list[float]:
    """Pad the box and its active visible points by 10%; clamp only this viewing crop."""
    x1, y1, x2, y2 = box.bbox
    if pose is not None and isinstance(box, PoseBox):
        for index in pose.class_keypoint_indices[box.class_id]:
            point = box.keypoints[index]
            if point.visibility == 2:
                x1, y1 = min(x1, point.x), min(y1, point.y)
                x2, y2 = max(x2, point.x), max(y2, point.y)
    dx, dy = (x2 - x1) * 0.1, (y2 - y1) * 0.1
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(1.0, x2 + dx), min(1.0, y2 + dy)]


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
    boxes: Sequence[Box],
    names: dict[int, str],
    max_edge: int,
    *,
    region: list[float] | None = None,
    encoding: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 95,
    coordinate_grid: bool = False,
    pose: PoseConfig | None = None,
    show_boxes: bool = True,
) -> tuple[list[JSONObject], JSONObject]:
    """Return raw/annotated views plus the actual rounded crop in full-image coordinates."""
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
    rendered = overlay(image, boxes, names, pose=pose, show_boxes=show_boxes, _viewport=extent)
    if region is not None:
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
        annotation = "keypoint IDs and names" if not show_boxes else "box IDs and classes"
        if show_boxes and pose is not None:
            annotation += ", keypoint IDs and names"
        caption = f"Same view with current {annotation}:"
        if coordinate_grid:
            caption += " Grid labels are FULL IMAGE normalized x/y."
        if pose is not None:
            caption += (
                " Point labels use object-ID/index:name; b1 abbreviates box_0001."
                " A ? means unreviewed; ~ and a hollow orange marker mean draft v=1."
            )
        content = [
            text_item("Unannotated image / region:"),
            image_item(raw, max_edge, encoding=encoding, jpeg_quality=jpeg_quality),
            text_item(caption),
            image_item(rendered, max_edge, encoding=encoding, jpeg_quality=jpeg_quality),
        ]
        return content, metadata
    finally:
        rendered.close()
        if raw is not image:
            raw.close()
