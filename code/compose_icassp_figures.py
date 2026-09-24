# -*- coding: utf-8 -*-
"""Rebuild the three ICASSP figures with embedded TrueType fonts."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.lib.colors import white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


FIG1_SIZE = (1175.793, 779.7629)
FIG2_SIZE = (767.995482, 418.187187)
FIG3_SIZE = (1019.27024, 656.829115)


def remove_unused_core_font(page) -> None:
    """Remove ReportLab's empty Helvetica setup and its unused resource."""
    data = page.get_contents().get_data()
    data = re.sub(rb"BT(?:(?!Tj|TJ).)*?ET\s*", b"", data, flags=re.DOTALL)
    contents = DecodedStreamObject()
    contents.set_data(data)
    page[NameObject("/Contents")] = contents

    fonts_reference = page["/Resources"].get("/Font")
    if fonts_reference is None:
        return
    fonts = fonts_reference.get_object()
    for resource_name, font_reference in list(fonts.items()):
        base_font = str(font_reference.get_object().get("/BaseFont", ""))
        if base_font in {"/Helvetica", "/Helvetica-Bold"}:
            del fonts[resource_name]


def overlay_page(
    size: tuple[float, float],
    labels: list[tuple[str, float, float, float]],
    image=None,
    white_background: bool = True,
):
    stream = BytesIO()
    drawing = canvas.Canvas(stream, pagesize=size)
    if white_background:
        drawing.setFillColor(white)
        drawing.rect(0, 0, size[0], size[1], stroke=0, fill=1)

    if image is not None:
        image_stream = BytesIO()
        image.save(image_stream, format="PNG")
        image_stream.seek(0)
        drawing.drawImage(
            ImageReader(image_stream),
            127.8967,
            20.0,
            width=920.0,
            height=257.0749,
            mask="auto",
        )

    drawing.setFillColorRGB(0, 0, 0)
    for text, x, y, font_size in labels:
        drawing.setFont("Arial-Bold-Embedded", font_size)
        drawing.drawString(x, y, text)

    drawing.save()
    stream.seek(0)
    page = PdfReader(stream).pages[0]
    remove_unused_core_font(page)
    return page


def merge_fitted_source(
    target_page,
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    source_page = PdfReader(path).pages[0]
    source_width = float(source_page.mediabox.width)
    source_height = float(source_page.mediabox.height)
    target_page.merge_transformed_page(
        source_page,
        Transformation()
        .scale(width / source_width, height / source_height)
        .translate(x, y),
    )


def write_page(page, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(page)
    with output_path.open("wb") as output:
        writer.write(output)


def build_fig1(reference_path: Path, waveform_path: Path, output_path: Path) -> None:
    reference = PdfReader(reference_path).pages[0]
    images = list(reference.images)
    if len(images) != 1:
        raise ValueError(f"Expected one network-diagram image in {reference_path}, found {len(images)}")

    page = overlay_page(FIG1_SIZE, [], image=images[0].image)
    merge_fitted_source(page, waveform_path, 22.0, 303.074911, 1131.793, 456.688)
    labels = overlay_page(
        FIG1_SIZE,
        [("A", 18.0, 750.7629, 32.0), ("B", 18.0, 257.0749, 32.0)],
        white_background=False,
    )
    page.merge_page(labels)
    write_page(page, output_path)


def build_fig2(boxplot_path: Path, confusion_path: Path, output_path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=FIG2_SIZE[0], height=FIG2_SIZE[1])
    merge_fitted_source(page, boxplot_path, 18.0, 14.0, 313.83, 370.187187)
    merge_fitted_source(page, confusion_path, 355.83, 14.0, 394.165482, 370.187187)
    labels = overlay_page(
        FIG2_SIZE,
        [("A", 18.0, 394.1872, 20.0), ("B", 355.83, 394.1872, 20.0)],
        white_background=False,
    )
    page.merge_page(labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        writer.write(output)


def copy_fig3(source_path: Path, output_path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=FIG3_SIZE[0], height=FIG3_SIZE[1])
    merge_fitted_source(page, source_path, 0.0, 0.0, FIG3_SIZE[0], FIG3_SIZE[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        writer.write(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-fig1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    pdfmetrics.registerFont(TTFont("Arial-Bold-Embedded", r"C:\Windows\Fonts\arialbd.ttf"))

    build_fig1(
        args.reference_fig1,
        project_root / "analysis" / "merkel_meissner_8materials.pdf",
        args.output_dir / "fig1.pdf",
    )
    build_fig2(
        project_root / "analysis" / "analysi.ipynb" / "boxplot_8cls_Tn25_STDP_T_STDP_SRDP.pdf",
        project_root / "analysis" / "analysi.ipynb" / "conf8cls_icann_bw.pdf",
        args.output_dir / "fig2.pdf",
    )
    copy_fig3(
        project_root
        / "pca_2d_results"
        / "integrated_pca_loadings"
        / "pca_scatter_pc1_pc2_loadings_rep1.pdf",
        args.output_dir / "fig3.pdf",
    )


if __name__ == "__main__":
    main()
