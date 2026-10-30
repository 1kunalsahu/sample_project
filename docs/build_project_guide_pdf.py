"""Build the project guide PDF using PyMuPDF (fitz)."""
from __future__ import annotations

import html
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "project_guide.md"
DESTINATION = ROOT / "docs" / "CriticRAG_Project_Guide.pdf"


def inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<code>\1</code>', escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_to_html(source: str) -> str:
    output: list[str] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        if paragraph:
            output.append("<p>" + " ".join(inline(x.strip()) for x in paragraph) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    for line in source.splitlines():
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code:
                output.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>")
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            continue
        if line.startswith("#"):
            flush_paragraph()
            close_list()
            level = min(len(line) - len(line.lstrip("#")), 3)
            title = line[level:].strip()
            output.append(f"<h{level}>{inline(title)}</h{level}>")
        elif line.startswith("- "):
            flush_paragraph()
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append("<li>" + inline(line[2:]) + "</li>")
        else:
            paragraph.append(line)
    flush_paragraph()
    close_list()
    if in_code and code_lines:
        output.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>")
    return "<html><body>" + "".join(output) + "</body></html>"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = markdown_to_html(source)
    css = """
      @font-face { font-family: Arial; src: url('C:/Windows/Fonts/arial.ttf'); }
      body { font-family: Arial, sans-serif; font-size: 10.3pt; color: #182333; }
      h1 { font-size: 23pt; color: #12345a; margin: 0 0 15pt; }
      h2 { font-size: 16pt; color: #12345a; margin: 18pt 0 7pt; }
      h3 { font-size: 12pt; color: #176b78; margin: 13pt 0 5pt; }
      p { line-height: 1.35; margin: 0 0 8pt; }
      ul { margin: 0 0 8pt 13pt; padding-left: 8pt; }
      li { line-height: 1.25; margin-bottom: 4pt; }
      code { font-family: Courier, monospace; font-size: 9pt; color: #874b16; }
      pre { font-family: Courier, monospace; font-size: 8.4pt; line-height: 1.25; background-color: #f1f4f8; padding: 8pt; }
    """
    story = fitz.Story(html=body, user_css=css)
    writer = fitz.DocumentWriter(str(DESTINATION))
    page = fitz.paper_rect("a4")
    area = fitz.Rect(48, 48, page.width - 48, page.height - 48)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
    writer.close()
    print(f"Created {DESTINATION}")


if __name__ == "__main__":
    main()
