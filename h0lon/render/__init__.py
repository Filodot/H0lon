"""Rendering: Pandoc Markdown master → PDF (XeLaTeX, fallback HTML → browser)."""

from h0lon.render.pdfcheck import PdfCheckReport, check_pdf
from h0lon.render.pipeline import RenderReport, render_document
from h0lon.render.templates import TemplateInfo, get_template, list_templates

__all__ = [
    "PdfCheckReport",
    "RenderReport",
    "TemplateInfo",
    "check_pdf",
    "get_template",
    "list_templates",
    "render_document",
]
