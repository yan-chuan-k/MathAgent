from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple
from xml.sax.saxutils import escape


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "math_agent_project_record.docx"


CODE_FILES = [
    "main.py",
    "user_agent.py",
    "intern_s1_client.py",
    "run_batch.py",
    "validate_results.py",
    "result_validator.py",
    "json_validator.py",
    "math_agent.py",
    "math_agent_core/__init__.py",
    "math_agent_core/orchestrator.py",
    "math_agent_core/prompts.py",
    "math_agent_core/json_utils.py",
    "math_agent_core/schema.py",
    "math_agent_core/answer_utils.py",
    "math_agent_core/trace_utils.py",
    "math_agent_core/clients/__init__.py",
    "math_agent_core/clients/mock_client.py",
    "test_intern_s1.py",
    "tests/test_answer_extraction.py",
    "tests/test_no_secret_leak.py",
    "tests/test_sample_runner.py",
    "tests/test_user_agent_entry.py",
    "result_schema.json",
    "requirements.txt",
    "README.md",
    "input.json",
    "problems.jsonl",
    "sample_data/dev.jsonl",
]


FILE_NOTES = {
    "main.py": "项目主入口。支持官方 baseline JSONL 批处理和旧版单题 JSON 两种运行方式，负责参数解析、路径解析、client 构建、并发执行和结果落盘。",
    "user_agent.py": "面向评测入口的 ReasoningAgent。将问题交给 MathAgentOrchestrator，抽取最终答案，并输出 final_response 与 trace。",
    "intern_s1_client.py": "Intern-S1/OpenAI 兼容接口封装。负责读取环境变量、初始化客户端、发送 chat 请求、重试和返回文本或 JSON。",
    "run_batch.py": "旧版批处理脚本。读取 problems.jsonl，调用 MathAgentOrchestrator，按 schema 分流有效/无效结果并写入日志和汇总。",
    "validate_results.py": "命令行 JSONL 校验工具。逐行读取结果并用 result_schema.json 验证。",
    "result_validator.py": "轻量 schema 校验函数，供代码调用。",
    "json_validator.py": "早期 JSON 校验类，部分中文提示存在编码异常，功能与 result_validator.py 有重叠。",
    "math_agent.py": "历史版本单 Agent 设计草稿。引用的 InternS1 类在当前代码中不存在，不建议作为当前入口。",
    "math_agent_core/orchestrator.py": "核心编排器。完成输入标准化、提示词调用、JSON 抽取/修复、schema 校验、重试、日志和最终结果组装。",
    "math_agent_core/prompts.py": "构造系统提示词和用户提示词，约束模型输出严格 JSON。",
    "math_agent_core/json_utils.py": "从模型文本中抽取 JSON、本地修复字段、调用 jsonschema 校验。",
    "math_agent_core/schema.py": "生成 schema 兼容的空结果和 _meta 元数据默认值。",
    "math_agent_core/answer_utils.py": "从结构化结果或文本中抽取最终答案，服务官方 final_response 输出。",
    "math_agent_core/trace_utils.py": "构造 trace 记录，将结构化求解结果转成可追踪步骤。",
    "math_agent_core/clients/mock_client.py": "离线测试替身，返回 schema 合法的模板结果，用于无 API 环境测试。",
    "result_schema.json": "统一输出结构约束，定义 problem_id、problem_type、solution、final_answer、verification、_meta 等必填字段。",
    "requirements.txt": "项目依赖清单。",
    "README.md": "项目运行说明。",
}


def read_text(relative_path: str) -> str:
    path = BASE_DIR / relative_path
    if not path.exists():
        return "[文件不存在]"
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(relative_path: str):
    path = BASE_DIR / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def paragraph(text: str = "", style: str | None = None) -> str:
    style_xml = ""
    if style:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    runs = []
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if index:
            runs.append("<w:r><w:br/></w:r>")
        escaped = escape(line)
        runs.append(f'<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r>')
    return f"<w:p>{style_xml}{''.join(runs)}</w:p>"


def code_paragraph(text: str) -> str:
    escaped = escape(text)
    return (
        '<w:p><w:pPr><w:pStyle w:val="Code"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'
    )


def bullet(text: str) -> str:
    return paragraph("• " + text)


def table(rows: Iterable[Tuple[str, str]]) -> str:
    xml = [
        "<w:tbl>",
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>",
    ]
    for left, right in rows:
        xml.append("<w:tr>")
        for value in (left, right):
            xml.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"4500\" w:type=\"dxa\"/></w:tcPr>"
                + paragraph(value)
                + "</w:tc>"
            )
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def section_title(text: str) -> str:
    return paragraph(text, "Heading1")


def sub_title(text: str) -> str:
    return paragraph(text, "Heading2")


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def build_document_body() -> str:
    validation_summary = load_json("validation_summary.json") or {}
    requirements = read_text("requirements.txt").strip()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: List[str] = []
    parts.append(paragraph("Math Agent 项目整体记录文档", "Title"))
    parts.append(paragraph(f"生成时间：{now}"))
    parts.append(paragraph(f"项目路径：{BASE_DIR}"))
    parts.append(
        paragraph(
            "文档目的：记录当前 math_agent 项目的整体内容、代码作用、实现思路、进度状态和完整源码，便于后续交给 GPT 进行分析研究。"
        )
    )

    parts.append(section_title("一、项目概览"))
    parts.append(
        paragraph(
            "该项目是一个数学题求解 Agent。核心目标是接收数学题文本，调用 Intern-S1 或离线 MockClient，要求模型按统一 JSON Schema 输出题型、推理计划、解题步骤、最终答案、验证过程和元数据。"
        )
    )
    parts.append(
        paragraph(
            "当前代码同时保留两套运行路径：一套是面向官方 baseline 的 main.py + ReasoningAgent 输出 final_response；另一套是 run_batch.py + MathAgentOrchestrator 的结构化 JSONL 批处理。"
        )
    )

    parts.append(section_title("二、当前进度"))
    progress_rows = [
        ("单元测试", "已执行 python -m pytest，结果为 8 passed, 1 skipped。"),
        (
            "结果校验",
            "validation_summary.json 显示 total=3, valid_count=3, invalid_count=0, failed_count=0, valid_rate=1.0。",
        ),
        ("接口状态", "支持真实 Intern-S1 API，也支持 --mock 离线测试。真实 API 需要 .env 中的 INTERN_API_KEY。"),
        ("安全处理", ".env、缓存、日志和模型输出未作为源码附录收录，避免密钥或临时数据进入文档。"),
    ]
    if validation_summary:
        progress_rows.append(("validation_summary.json", json.dumps(validation_summary, ensure_ascii=False)))
    parts.append(table(progress_rows))

    parts.append(section_title("三、整体思路"))
    parts.append(bullet("输入层：main.py 或 run_batch.py 读取 JSON/JSONL，将题目文本和 problem_id 标准化。"))
    parts.append(bullet("模型层：InternS1Client 封装 OpenAI 兼容 chat.completions 接口；MockClient 用于无网络、无密钥时测试。"))
    parts.append(bullet("编排层：MathAgentOrchestrator 构造提示词，调用模型，抽取 JSON，补齐缺失字段，执行 schema 校验。"))
    parts.append(bullet("修复层：如果 schema 不合法、验证失败、置信度低或答案为空，编排器会按 max_retries 重试。"))
    parts.append(bullet("输出层：结构化结果写入 JSON/JSONL；官方入口进一步抽取 final_response，并保留 trace 便于分析。"))
    parts.append(bullet("验证层：result_schema.json 统一约束输出格式，validate_results.py 和测试用例保障格式与密钥安全。"))

    parts.append(section_title("四、架构流程"))
    parts.append(
        paragraph(
            "推荐理解顺序：main.py -> user_agent.py -> math_agent_core/orchestrator.py -> prompts.py -> intern_s1_client.py/mock_client.py -> json_utils.py/schema.py -> answer_utils.py/trace_utils.py -> result_schema.json。"
        )
    )
    parts.append(
        paragraph(
            "典型执行流程：读取题目 -> 构造 ReasoningAgent 或 MathAgentOrchestrator -> 生成严格 JSON 提示词 -> 调用模型 -> 抽取/修复 JSON -> schema 校验 -> 生成 final_answer/final_response -> 保存结果和日志。"
        )
    )

    parts.append(section_title("五、文件作用说明"))
    rows = [("文件", "作用")]
    for filename in CODE_FILES:
        if (BASE_DIR / filename).exists():
            rows.append((filename, FILE_NOTES.get(filename, "项目文件，完整内容见源码附录。")))
    parts.append(table(rows))

    parts.append(section_title("六、运行方式"))
    parts.append(sub_title("官方 baseline 批处理"))
    parts.append(code_paragraph("python main.py --input_file sample_data\\dev.jsonl --output_dir sample_outputs --mock"))
    parts.append(sub_title("旧版单题运行"))
    parts.append(code_paragraph("python main.py --input input.json --output result.json --mock"))
    parts.append(sub_title("旧版 JSONL 批处理"))
    parts.append(code_paragraph("python run_batch.py --input problems.jsonl --output results.jsonl --mock"))
    parts.append(sub_title("结果校验"))
    parts.append(code_paragraph("python validate_results.py results.jsonl result_schema.json"))
    parts.append(sub_title("测试"))
    parts.append(code_paragraph("python -m pytest"))

    parts.append(section_title("七、依赖"))
    parts.append(code_paragraph(requirements or "[requirements.txt 为空]"))

    parts.append(section_title("八、当前风险和后续研究点"))
    parts.append(bullet("math_agent.py 是历史草稿，引用的 InternS1 类当前不存在，应避免作为主入口分析。"))
    parts.append(bullet("json_validator.py 和 validate_results.py 中存在中文乱码提示，不影响核心逻辑，但影响可读性。"))
    parts.append(bullet("MockClient 只保证流程和 schema 可测，不计算真实数学答案。真实效果需要用 Intern-S1 API 验证。"))
    parts.append(bullet("当前重试策略是重新请求模型，没有将上一轮错误细节明确加入下一轮 prompt，可作为后续优化点。"))
    parts.append(bullet("schema 校验主要保证结构合法，不保证数学答案正确；数学正确性依赖模型自检和后续人工/工具验证。"))

    parts.append(page_break())
    parts.append(section_title("九、完整源码附录"))
    for filename in CODE_FILES:
        path = BASE_DIR / filename
        if not path.exists():
            continue
        parts.append(sub_title(filename))
        note = FILE_NOTES.get(filename)
        if note:
            parts.append(paragraph("作用：" + note))
        content = read_text(filename)
        for line in content.splitlines() or [""]:
            parts.append(code_paragraph(line))

    return "".join(parts)


def write_docx(output_path: Path) -> None:
    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
<w:body>{build_document_body()}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1080" w:bottom="1440" w:left="1080" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body>
</w:document>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="40"/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="24"/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:eastAsia="Microsoft YaHei" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    document_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/_rels/document.xml.rels", document_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)


if __name__ == "__main__":
    write_docx(OUTPUT_PATH)
    print(OUTPUT_PATH)
