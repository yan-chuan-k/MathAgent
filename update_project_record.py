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
        "2026-08-01 更新记录：新增 18 类高难度诊断题集与诊断脚本。",
        "诊断题集：sample_data/hard_diagnostics.jsonl 覆盖离散数学、数值分析、测度积分、微分几何、概率论、抽象代数、随机过程、复分析、ODE、统计推断、泛函分析、线性回归、PDE、进阶课程、高等代数、运筹学、数学分析、拓扑学。",
        "诊断脚本：diagnose_hard_cases.py 支持路由检测、mock 管线检测和真实 API 模型检测；不会把 answer_hint 传入 agent。",
        "离线结果：当前无 INTERN_API_KEY，无法真实评估模型答案正确性；已完成路由检测 18/18 命中，mock ReasoningAgent 管线 18/18 成功。",
        "文档更新：README 修复乱码中文，加入 hard diagnostics 运行命令、当前离线结果和真实模型评测说明。",
    ]
    backup = append_docx_paragraphs(path, paragraphs)
    print(f"updated {path}; backup {backup}")


if __name__ == "__main__":
    main()
