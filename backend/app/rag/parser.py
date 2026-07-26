"""Document parser: PDF / Markdown / TXT / DOCX / 图片(OCR)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from loguru import logger


class DocumentParser:
    SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".docx", ".png", ".jpg", ".jpeg"}

    def parse(self, file_path: str) -> str:
        p = Path(file_path)
        suffix = p.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}")
        try:
            if suffix == ".pdf":
                return self._parse_pdf(p)
            if suffix in {".md", ".markdown"}:
                return p.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".txt":
                return p.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".docx":
                return self._parse_docx(p)
            if suffix in {".png", ".jpg", ".jpeg"}:
                return self._parse_image_ocr(p)
        except Exception as e:
            logger.exception(f"Parse failed: {file_path}: {e}")
            raise
        return ""

    def _parse_pdf(self, p: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
        return "\n\n".join(texts)

    def _parse_docx(self, p: Path) -> str:
        from docx import Document
        doc = Document(str(p))
        return "\n\n".join(par.text for par in doc.paragraphs if par.text.strip())

    def _parse_image_ocr(self, p: Path) -> str:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(str(p))
            return pytesseract.image_to_string(img, lang="chi_sim+eng")
        except Exception as e:
            logger.warning(f"OCR unavailable: {e}; return empty")
            return ""
