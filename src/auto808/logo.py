"""Lucid Triangulation Records logo generator.

Draws the geometric crest (three offset triangles, white core) plus
wordmark as a 1000x1000 SVG. Needs: svgwrite.
"""

from __future__ import annotations

import math

import svgwrite


def _triangle_vertices(center_x: float, center_y: float, radius: float,
                       angle_offset_deg: float):
    points = []
    for i in range(3):
        angle_rad = math.radians(angle_offset_deg + i * 120)
        x = center_x + radius * math.cos(angle_rad)
        y = center_y + radius * math.sin(angle_rad)
        points.append((x, y))
    return points


def create_lucid_triangulation_logo(
        filename: str = "lucid_triangulation_logo.svg") -> str:
    size = 1000
    dwg = svgwrite.Drawing(filename, size=(size, size), profile="full")

    COLOR_BG = "#0D0E12"
    COLOR_RED = "#E63946"
    COLOR_BLUE = "#005596"
    COLOR_PURPLE = "#6A0DAD"
    COLOR_WHITE = "#FFFFFF"

    dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill=COLOR_BG))

    cx, cy = 500, 380
    r = 150
    stroke_width = 14
    offset_distance = 45

    centers = [
        (cx + offset_distance * math.cos(math.radians(-90)),
         cy + offset_distance * math.sin(math.radians(-90))),
        (cx + offset_distance * math.cos(math.radians(30)),
         cy + offset_distance * math.sin(math.radians(30))),
        (cx + offset_distance * math.cos(math.radians(150)),
         cy + offset_distance * math.sin(math.radians(150))),
    ]

    triangles_data = [
        (_triangle_vertices(centers[0][0], centers[0][1], r, -90), COLOR_RED),
        (_triangle_vertices(centers[1][0], centers[1][1], r, -90), COLOR_BLUE),
        (_triangle_vertices(centers[2][0], centers[2][1], r, -90), COLOR_PURPLE),
    ]

    logo_mark = dwg.g(id="geometric_crest")
    for points, color in triangles_data:
        logo_mark.add(dwg.polygon(
            points=points, fill="none", stroke=color,
            stroke_width=stroke_width, stroke_linejoin="round", opacity=0.85))

    inner_core = _triangle_vertices(cx, cy - 10, r * 0.22, -90)
    logo_mark.add(dwg.polygon(points=inner_core, fill=COLOR_WHITE, opacity=0.95))
    dwg.add(logo_mark)

    dwg.add(dwg.text(
        "LUCID TRIANGULATION", insert=(cx, 710), text_anchor="middle",
        fill=COLOR_WHITE, font_size="38px",
        font_family="Montserrat, Helvetica, Arial, sans-serif",
        font_weight="bold", letter_spacing="12px"))
    dwg.add(dwg.text(
        "RECORDS", insert=(cx, 765), text_anchor="middle",
        fill="#ADAFAF", font_size="22px",
        font_family="Montserrat, Helvetica, Arial, sans-serif",
        font_weight="300", letter_spacing="18px"))

    dwg.save()
    print(f"Generated: {filename}")
    return filename
