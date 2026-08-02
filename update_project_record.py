from __future__ import annotations

import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}


def append_docx_paragraphs(path: Path, paragraphs: list[str]) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a valid docx zip package")

    backup = path.with_suffix(".docx.bak")
    shutil.copy2(path, backup)

    ET.register_namespace("w", WORD_NS)
    with zipfile.ZipFile(path, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    document_xml = "word/document.xml"
    root = ET.fromstring(files[document_xml])
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("word/document.xml has no w:body")

    section = body.find("w:sectPr", NS)
    children = list(body)
    insert_at = children.index(section) if section is not None else len(children)

    for offset, text in enumerate(paragraphs):
        body.insert(insert_at + offset, _paragraph(text))

    files[document_xml] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    tmp = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)
    try:
        tmp.replace(path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        tmp.replace(fallback)
        print(f"{path} is locked; wrote updated copy to {fallback}")
    return backup


def _paragraph(text: str) -> ET.Element:
    w = f"{{{WORD_NS}}}"
    paragraph = ET.Element(w + "p")
    run = ET.SubElement(paragraph, w + "r")
    text_node = ET.SubElement(run, w + "t")
    text_node.text = text
    return paragraph


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("math_agent_project_record.docx")
    paragraphs = [
        "2026-08-02 Phase 4B Linear Algebra Update: added matrix tool verification and whitelist registry.",
        "Added MathTool base and ToolRegistry for controlled whitelist tool execution.",
        "Added MatrixTool with bounded exact checks for determinants, matrix products, inverses, linear-system residuals, ranks, eigenpair residuals, orthogonality, normalization, and matrix/vector equivalence.",
        "Added math_agent_core/verifiers/linear_algebra.py. It consumes only structured requested_checks and never executes model-generated Python code.",
        "Extended requested_checks schema and normalization to preserve nested matrix/vector arguments safely.",
        "Integrated linear algebra evidence into MathAgentOrchestrator. Decisive matrix failures reject candidates; decisive matrix passes can satisfy AcceptancePolicy.",
        "Validation: python -m pytest -q => 47 passed, 1 skipped; python -m compileall -q .; mock baseline runner passed.",
    ]
    backup = append_docx_paragraphs(path, paragraphs)
    print(f"updated {path}; backup {backup}")


if __name__ == "__main__":
    main()
