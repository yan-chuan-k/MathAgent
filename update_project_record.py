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
        "2026-08-02 Phase 2 Update: added strategy-aware multi-candidate search and separated model roles.",
        "Added CandidateSolution, strategy pools, candidate ranker, and answer equivalence clustering in math_agent_core/candidate.py, math_agent_core/search/, and math_agent_core/evaluation/.",
        "Split prompt builders into Planner, Solver, Critic, Reviser, and Finalizer roles while preserving the existing solver output schema.",
        "MathAgentOrchestrator now generates candidates by strategy, validates each candidate with tools, optionally asks an independent Critic, clusters/ranks candidates, and selects the highest scoring verified answer.",
        "Finalizer support is optional and only formats the system-selected candidate into final_response; official ReasoningAgent keeps it disabled by default to avoid extra API cost.",
        "Extended ScriptedClient with role-aware response queues and added tests for candidate ranking, Critic rejection, and Finalizer formatting.",
        "Validation: python -m pytest -q => 32 passed, 1 skipped; python -m compileall -q .; mock baseline runner passed.",
    ]
    backup = append_docx_paragraphs(path, paragraphs)
    print(f"updated {path}; backup {backup}")


if __name__ == "__main__":
    main()
