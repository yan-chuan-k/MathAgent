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
        "2026-08-01 更新记录：使用 .env 中的 Intern-S API key 运行真实高难诊断，并修复发现的问题。",
        "真实诊断：18 个高难诊断题均返回非空、JSON 可序列化 final_response；路由仍为 18/18 命中。",
        "问题修复：diagnose_hard_cases.py 在 Windows 控制台打印数学 Unicode 时可能触发 UnicodeEncodeError，已加入安全打印兜底。",
        "诊断题修正：修正测度积分、复分析、常微分方程三道题的 answer_hint，避免后续评估参考答案误导。",
        "答案补救：微分几何题暴露模型把 K=1 写入 verification 但 final_response 漏主值的问题；ReasoningAgent 已加入保守补救，仅在题目询问高斯曲率且 verification/solution 中存在 K=...=数值时补到 final_response 开头。",
        "文档更新：README 记录 .env 真实诊断结果、Unicode 打印修复、诊断题答案提示修正和高斯曲率补救逻辑。",
    ]
    backup = append_docx_paragraphs(path, paragraphs)
    print(f"updated {path}; backup {backup}")


if __name__ == "__main__":
    main()
