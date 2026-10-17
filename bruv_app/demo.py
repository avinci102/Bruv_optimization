"""Synthetic data so the whole workflow runs without real footage or model weights.

- a mock Excel template with the observed sheet names, headers, hidden columns and
  dropdowns (some stored in Excel's x14 extension, like real cross-sheet dropdowns);
- a two-chapter "GoPro" video of coloured fish over murky water;
- a demo classifier that "knows" the fish colours.

Used by the tests and by `python -m bruv_app demo`.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image, ImageDraw

from .classify import histogram_embed, l2_normalize
from .template import sheet_xml_paths

RED = "Lutjanus campechanus"
YELLOW = "Seriola dumerili"
GREEN = "Lachnolaimus maximus"
COLORS = {RED: (215, 45, 35), YELLOW: (230, 200, 40), GREEN: (50, 185, 70)}
OTHER_SPECIES = ["Balistes capriscus", "Rachycentron canadum", "Mycteroperca microlepis"]
WATER = (62, 82, 96)

SIZE = (640, 360)
SOURCE_FPS = 10
CHAPTER_S = 12.0

SHEET_DROPDOWN = "dropdownlist (dont touch this)"
SHEET_INFO = "info (fill this first)"
SHEET_RECORD = "record (fill this next)"
# A-F are unknown in the real file; these placeholders stand in for them.
RECORD_HEADERS = {
    "A": "opcode", "B": "site", "C": "date", "D": "depth", "E": "analyst", "F": "video",
    "G": "clip_in", "H": "time_in", "I": "clip_out", "J": "time_out",
    "K": "hidden_k", "L": "hidden_l",
    "M": "type", "N": "species", "O": "count",
    "P": "hidden_p", "Q": "hidden_q",
    "R": "length_ind", "S": "school_min", "T": "school_max",
    "U": "hidden_u",
    "V": "sex", "W": "behavior", "X": "comment",
}
HIDDEN = ["K", "L", "P", "Q", "U"]
TYPES = ["individual", "school", "other"]
SEXES = ["M", "F", "U"]
BEHAVIORS = ["feeding", "passing", "bait interaction", "schooling"]


@dataclass
class SceneFish:
    species: str
    t0: float
    t1: float
    x0: float  # centre x at t0, pixels
    y: float
    vx: float  # pixels per second
    length: float = 90.0

    def box(self, t: float) -> tuple[float, float, float, float] | None:
        if not self.t0 <= t <= self.t1:
            return None
        cx = self.x0 + self.vx * (t - self.t0)
        cy = self.y + 6 * np.sin(t)
        half_l, half_h = self.length / 2, self.length * 0.22
        return cx - half_l, cy - half_h, cx + half_l, cy + half_h


# Red visits twice (3 s gap -> one review card), a yellow crosses the chapter boundary,
# a yellow school of 3 gives MaxN 3, two reds together give MaxN 2.
SCENE = [
    SceneFish(RED, 0.5, 5.2, 80, 90, 55),
    SceneFish(RED, 8.2, 11.4, 520, 100, -50),
    SceneFish(YELLOW, 2.5, 6.8, 560, 250, -60),
    SceneFish(YELLOW, 9.6, 14.2, 100, 300, 45),
    SceneFish(YELLOW, 15.5, 20.4, 90, 60, 40),
    SceneFish(YELLOW, 15.5, 20.4, 140, 160, 40),
    SceneFish(YELLOW, 15.5, 20.4, 100, 260, 40),
    SceneFish(RED, 16.3, 18.6, 560, 110, -40),
    SceneFish(RED, 16.3, 18.6, 520, 220, -40),
    SceneFish(GREEN, 20.5, 23.5, 300, 180, 30),
]


def scene_at(t: float) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Ground truth: (species, normalised box) for each fish visible at time t."""
    w, h = SIZE
    out = []
    for f in SCENE:
        b = f.box(t)
        if b is not None:
            out.append((f.species, (b[0] / w, b[1] / h, b[2] / w, b[3] / h)))
    return out


def _draw_fish(draw: ImageDraw.ImageDraw, box, color, facing_right: bool) -> None:
    x1, y1, x2, y2 = box
    length = x2 - x1
    tail = length * 0.25
    if facing_right:
        draw.ellipse([x1 + tail, y1, x2, y2], fill=color)
        draw.polygon([(x1 + tail + 4, (y1 + y2) / 2), (x1, y1), (x1, y2)], fill=color)
    else:
        draw.ellipse([x1, y1, x2 - tail, y2], fill=color)
        draw.polygon([(x2 - tail - 4, (y1 + y2) / 2), (x2, y1), (x2, y2)], fill=color)


def render_frame(t: float, rng: np.random.Generator) -> np.ndarray:
    w, h = SIZE
    base = np.empty((h, w, 3), np.float32)
    base[:] = WATER
    base += np.linspace(-8, 8, h)[:, None, None]  # light from above
    base += rng.normal(0, 3, (h, w, 1))
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for f in SCENE:
        b = f.box(t)
        if b is not None:
            _draw_fish(draw, b, COLORS[f.species], f.vx > 0)
    return np.asarray(image)


def make_synthetic_videos(folder: Path, chapters: int = 2) -> list[Path]:
    """Write GoPro-style chapter files (GX01xxxx, GX02xxxx). Returns them out of order."""
    import imageio_ffmpeg  # noqa: PLC0415

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for c in range(chapters):
        path = folder / f"GX{c + 1:02d}0042.MP4"
        writer = imageio_ffmpeg.write_frames(
            str(path), SIZE, fps=SOURCE_FPS, codec="libx264", macro_block_size=8, quality=8
        )
        writer.send(None)
        n = int(CHAPTER_S * SOURCE_FPS)
        for k in range(n):
            writer.send(render_frame(c * CHAPTER_S + k / SOURCE_FPS, rng))
        writer.close()
        paths.append(path)
    return list(reversed(paths))


def make_mock_template(path: Path) -> Path:
    """Mock of the owner's viewing template, for tests and the demo only."""
    wb = openpyxl.Workbook()
    dd = wb.active
    dd.title = SHEET_DROPDOWN
    species = [RED, YELLOW, GREEN, *OTHER_SPECIES]
    for col, (header, values) in enumerate(
        [("species", species), ("type", TYPES), ("sex", SEXES), ("behavior", BEHAVIORS)], start=1
    ):
        dd.cell(1, col, header)
        for r, v in enumerate(values, start=2):
            dd.cell(r, col, v)

    info = wb.create_sheet(SHEET_INFO)
    for r, field in enumerate(["opcode", "date", "site", "latitude", "longitude", "depth_m", "analyst"], 1):
        info.cell(r, 1, field)

    rec = wb.create_sheet(SHEET_RECORD)
    for letter, header in RECORD_HEADERS.items():
        rec[f"{letter}1"] = header
    for letter in HIDDEN:
        rec.column_dimensions[letter].hidden = True
    rec["G2"], rec["I2"] = "clip_1", "clip_1"
    for cell in ("H2", "J2"):
        rec[cell] = 0.0
        rec[cell].number_format = "hh:mm:ss"

    # Standard validations: a named range (type) and a literal list (sex).
    wb.defined_names["type_list"] = DefinedName("type_list", attr_text=f"'{SHEET_DROPDOWN}'!$B$2:$B${len(TYPES) + 1}")
    dv = DataValidation(type="list", formula1="type_list", allow_blank=True)
    dv.add("M2:M1000")
    rec.add_data_validation(dv)
    dv = DataValidation(type="list", formula1='"M,F,U"', allow_blank=True)
    dv.add("V2:V1000")
    rec.add_data_validation(dv)

    buf = io.BytesIO()
    wb.save(buf)
    # Cross-sheet lists (species, behavior) go in the x14 extension, as Excel 2010+ writes them.
    ext = (
        '<extLst><ext uri="{CCE6A557-97BC-4b89-ADB6-D9C93CAAB3DF}" '
        'xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main">'
        '<x14:dataValidations count="2" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main">'
        f'<x14:dataValidation type="list" allowBlank="1" showInputMessage="1" showErrorMessage="1">'
        f"<x14:formula1><xm:f>'{SHEET_DROPDOWN}'!$A$2:$A$50</xm:f></x14:formula1>"
        "<xm:sqref>N2:N1000</xm:sqref></x14:dataValidation>"
        f'<x14:dataValidation type="list" allowBlank="1" showInputMessage="1" showErrorMessage="1">'
        f"<x14:formula1><xm:f>'{SHEET_DROPDOWN}'!$D$2:$D$50</xm:f></x14:formula1>"
        "<xm:sqref>W2:W1000</xm:sqref></x14:dataValidation>"
        "</x14:dataValidations></ext></extLst>"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        record_part = sheet_xml_paths(src)[SHEET_RECORD]
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == record_part:
                data = data.decode().replace("</worksheet>", ext + "</worksheet>").encode()
            dst.writestr(item, data)
    return path


class DemoClassifier:
    """Histogram embeddings plus label 'text' embeddings rendered from the known colours."""

    name = "demo-colour"
    logit_scale = 12.0

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        return histogram_embed(crops)

    def text_embeddings(self, labels: list[str]) -> np.ndarray | None:
        rng = np.random.default_rng(1)
        out = []
        for label in labels:
            if label in COLORS:
                patch = Image.new("RGB", (100, 44), WATER)
                _draw_fish(ImageDraw.Draw(patch), (5, 2, 95, 42), COLORS[label], True)
                out.append(histogram_embed([np.asarray(patch)])[0])
            else:
                out.append(l2_normalize(rng.random(64)) * 0.2)
        return l2_normalize(np.array(out))
