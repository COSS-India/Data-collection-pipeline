from pathlib import Path

from dcp.extract import register
from dcp.models import TextBlock


@register("docx", ".docx")
def extract_docx(path: Path) -> list[TextBlock]:
    import docx

    d = docx.Document(str(path))
    blocks = [TextBlock(p.text, f"para={i}") for i, p in enumerate(d.paragraphs, 1)]
    for t, table in enumerate(d.tables, 1):
        for r, row in enumerate(table.rows, 1):
            seen = set()
            for c, cell in enumerate(row.cells, 1):
                if cell._tc in seen:  # merged cells repeat
                    continue
                seen.add(cell._tc)
                blocks.append(TextBlock(cell.text, f"table={t}:row={r}:col={c}"))
    return blocks


@register("pptx", ".pptx")
def extract_pptx(path: Path) -> list[TextBlock]:
    from pptx import Presentation

    blocks = []
    for s, slide in enumerate(Presentation(str(path)).slides, 1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs)
                    blocks.append(TextBlock(text, f"slide={s}"))
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        blocks.append(TextBlock(cell.text, f"slide={s}:table"))
        if slide.has_notes_slide:
            blocks.append(TextBlock(slide.notes_slide.notes_text_frame.text, f"slide={s}:notes"))
    return blocks


@register("xlsx", ".xlsx", ".xlsm")
def extract_xlsx(path: Path) -> list[TextBlock]:
    from openpyxl import load_workbook

    from dcp.extract.text import _is_texty

    wb = load_workbook(str(path), read_only=True, data_only=True)
    blocks = []
    for ws in wb.worksheets:
        for r, row in enumerate(ws.iter_rows(values_only=True), 1):
            for c, val in enumerate(row, 1):
                if isinstance(val, str) and _is_texty(val):
                    blocks.append(TextBlock(val, f"sheet={ws.title}:row={r}:col={c}"))
    return blocks


@register("odt", ".odt", ".odp")
def extract_odf(path: Path) -> list[TextBlock]:
    from odf import teletype, text as odftext
    from odf.opendocument import load

    doc = load(str(path))
    return [
        TextBlock(teletype.extractText(p), f"para={i}")
        for i, p in enumerate(doc.getElementsByType(odftext.P), 1)
    ]
