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
        "2026-08-02 Update: completed phase-1 correctness architecture upgrade for high-difficulty math solving.",
        "Added system-owned solve status fields: schema_valid, content_complete, answer_verified, proof_verified, overall_status, failure_kind, and failure_details. Model-provided _meta is ignored.",
        "Added VerificationEvidence and SolveAssessment in math_agent_core/state.py, plus safe whitelist SymPy verification in math_agent_core/tools/sympy_tool.py for arithmetic, equation substitution, symbolic equivalence, derivative checks, and integral checks.",
        "Changed MathAgentOrchestrator retries from repeated identical prompts to targeted repair prompts carrying schema errors, residuals, previous answers, and concrete verifier evidence.",
        "Hardened ReasoningAgent: unverified raw model output can no longer become final_response; only system-accepted solved candidates are returned, otherwise the conservative fallback is used with trace evidence.",
        "Added ScriptedClient and FaultInjectionClient tests for invalid JSON, wrong-answer rejection, repair feedback, inconclusive tools, and model _meta forgery.",
        "Hardened main.py resume behavior: shared local client, skip only successful non-empty outputs, and atomic JSON writes. Added Retry-After and jitter support in intern_s1_client.py without logging secrets.",
        "Validation: import ok; python -m pytest -q => 29 passed, 1 skipped; python -m compileall -q .; mock baseline runner passed.",
    ]
    backup = append_docx_paragraphs(path, paragraphs)
    print(f"updated {path}; backup {backup}")


if __name__ == "__main__":
    main()
