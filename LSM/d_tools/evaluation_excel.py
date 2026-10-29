"""参照ブックと同じ3シート形式で分類結果を保存・読み込みする。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np


def _dataset_name(source):
    """結果階層から学習則名を取り、accuracyシートのdataset欄に使う。"""
    # 出力層なしの結果にも、accuracyシートで正しいデータ名を付ける。
    if any(parent.name == "liquid_run" for parent in Path(source).resolve().parents):
        return "liquid"
    # off/repN/testと、STDP/パラメータ/repN/testの双方に対応する。
    for parent in Path(source).resolve().parents:
        if parent.name in {"off", "off2", "STDP", "T_STDP", "SRDP", "BCPNN"}:
            return parent.name
    return Path(source).parent.name


def save_classification_workbook(
    output,
    *,
    materials,
    summaries,
    details,
    matrices,
    texture_summaries,
    texture_details,
    texture_matrices,
    source,
    folds,
    base_seed,
):
    """参照と同じconf_8cls・conf_3cls・accuracyの3シートへ保存する。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = Path(
        os.environ.get(
            "LSM_DOCUMENT_RUNTIME",
            str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"),
        )
    )
    node = runtime / "node/bin/node.exe"
    modules = runtime / "node/node_modules"
    template = Path(__file__).with_name("classification_template.xlsx")
    if not node.is_file() or not (modules / "@oai/artifact-tool").is_dir():
        raise RuntimeError(
            "Excel export needs Node.js and @oai/artifact-tool in LSM_DOCUMENT_RUNTIME"
        )
    if not template.is_file():
        raise FileNotFoundError(f"Classification Excel template not found: {template}")

    # repと時間窓で対応を取り、8素材と3種類の結果を同じ行にまとめる。
    def key(row):
        """同じ評価条件の行・混同行列を対応付ける。"""
        return int(row["rep"]), float(row["T_n_ms"])

    records8 = {key(row): row for row in matrices}
    records3 = {key(row): row for row in texture_matrices}
    summaries3 = {key(row): row for row in texture_summaries}
    accuracy_rows = [
        [
            "dataset",
            "rep",
            "T_n",
            "accuracy8_overall",
            "accuracy8_mean",
            "accuracy3_overall",
            "accuracy3_mean",
            "n_sample_per_class",
            "total_samples",
        ]
    ]
    blocks8, blocks3 = [], []
    for index, summary in enumerate(summaries):
        condition = key(summary)
        cm8 = np.asarray(records8[condition]["counts"], dtype=np.int64)
        cm3 = np.asarray(records3[condition]["counts"], dtype=np.int64)
        if cm8.shape != (8, 8) or cm3.shape != (3, 3) or len(materials) != 8:
            raise ValueError(
                "The reference Excel format requires 8 materials and 3 texture classes"
            )
        support = cm8.sum(axis=1)
        if not np.all(support == support[0]) or cm8.sum() != cm3.sum():
            raise ValueError(
                "Expected equal trial counts per material and matching 8/3-class totals"
            )
        # 通常の1条件は参照と同じセル配置。複数時間窓は同じシートの下に追加する。
        blocks8.append(
            {
                "row": index * 12,
                "rows": [[None, *materials]]
                + [[label, *row.tolist()] for label, row in zip(materials, cm8)],
            }
        )
        blocks3.append({"row": index * 7, "rows": [[0, 1, 2], *cm3.tolist()]})
        summary3 = summaries3[condition]
        accuracy_rows.append(
            [
                _dataset_name(source),
                condition[0],
                summary["T_n_ms"],
                summary["accuracy"],
                summary["mean_fold_accuracy"],
                summary3["accuracy"],
                summary3["mean_fold_accuracy"],
                int(support[0]),
                int(cm8.sum()),
            ]
        )

    # 評価条件と分割別結果はシートを増やさず、ブック内の定義名へ格納する。
    # 再開時にseed等を照合でき、別のJSONやNPZファイルを作る必要がない。
    metadata = {
        "format_version": 2,
        "source": str(Path(source).resolve()),
        "materials": list(materials),
        "folds": folds,
        "base_seed": base_seed,
        "T_n_ms": list(dict.fromkeys(row["T_n_ms"] for row in summaries)),
        "summary": summaries,
        "texture_summary": texture_summaries,
        "details": details,
        "texture_details": texture_details,
    }
    workbook_path = (output / "classification.xlsx").resolve()
    books = [
        {
            "path": str(workbook_path),
            "template": str(template.resolve()),
            "sheets": [
                {"name": "conf_8cls", "blocks": blocks8},
                {"name": "conf_3cls", "blocks": blocks3},
                {"name": "accuracy", "blocks": [{"row": 0, "rows": accuracy_rows}]},
            ],
            "metadata": metadata,
        }
    ]
    subprocess.run(
        [str(node), str(Path(__file__).with_suffix(".mjs")), "--stdin", str(modules)],
        input=json.dumps(books, ensure_ascii=False, allow_nan=False),
        text=True,
        encoding="utf-8",
        check=True,
    )
    return workbook_path


def load_classification_results(path):
    """再開判定に必要な評価条件と正答率を、Excel自体から読み出す。"""
    import posixpath
    from zipfile import ZipFile, BadZipFile
    from xml.etree import ElementTree as ET

    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = [
                    "".join(node.itertext())
                    for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))
                ]
            relations = {
                r.attrib["Id"]: r.attrib["Target"]
                for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            rows_by_sheet = {}
            for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall(
                "s:sheets/s:sheet", ns
            ):
                rid = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                target = relations[rid]
                target = (
                    target.lstrip("/")
                    if target.startswith("/")
                    else posixpath.normpath(posixpath.join("xl", target))
                )
                rows = []
                for row in ET.fromstring(archive.read(target)).findall("s:sheetData/s:row", ns):
                    values = []
                    for cell in row.findall("s:c", ns):
                        col = 0
                        for ch in cell.attrib["r"]:
                            if not ch.isalpha():
                                break
                            col = col * 26 + ord(ch.upper()) - 64
                        while len(values) < col:
                            values.append(None)
                        value = cell.find("s:v", ns)
                        text = value.text if value is not None else None
                        kind = cell.get("t")
                        if kind == "s":
                            parsed = shared[int(text)]
                        elif kind == "inlineStr":
                            parsed = "".join(cell.find("s:is", ns).itertext())
                        elif kind == "b":
                            parsed = text == "1"
                        elif text is None:
                            parsed = None
                        elif kind in ("str", "e"):
                            parsed = text
                        else:
                            parsed = float(text)
                        values[col - 1] = parsed
                    rows.append(values)
                rows_by_sheet[sheet.attrib["name"]] = rows
        # 新形式は定義名の文字列定数から評価条件を復元する。
        import base64

        with ZipFile(path) as archive:
            workbook_xml = ET.fromstring(archive.read("xl/workbook.xml"))
        chunks = sorted(
            (name.get("name"), name.text)
            for name in workbook_xml.findall("s:definedNames/s:definedName", ns)
            if name.get("name", "").startswith("_LSM_EVAL_")
        )
        if chunks:
            encoded = "".join(text.lstrip("=").strip('"') for _, text in chunks)
            result = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
            if result.get("format_version") != 2:
                return None
            expected_headers = [
                "dataset",
                "rep",
                "T_n",
                "accuracy8_overall",
                "accuracy8_mean",
                "accuracy3_overall",
                "accuracy3_mean",
                "n_sample_per_class",
                "total_samples",
            ]
            if list(rows_by_sheet) != ["conf_8cls", "conf_3cls", "accuracy"]:
                return None
            if rows_by_sheet["accuracy"][0] != expected_headers:
                return None
            return result
        # 従来形式のExcelも読めるようにして、既存結果の再開・移行を可能にする。
        conditions = dict(rows_by_sheet["評価条件"][1:])
        if conditions.get("出力形式バージョン") != 1:
            return None
        fields = ("rep", "T_n_ms", "folds", "accuracy", "mean_fold_accuracy", "std_fold_accuracy")

        def summaries(name):
            return [dict(zip(fields, row)) for row in rows_by_sheet[name][1:]]

        result = {
            "source": conditions["記録データ"],
            "materials": json.loads(conditions["素材順"]),
            "folds": conditions["交差検証の分割数"],
            "base_seed": conditions["評価乱数seed"],
            "T_n_ms": json.loads(conditions["集約時間窓 (ms)"]),
            "summary": summaries("8素材_正答率"),
            "texture_summary": summaries("3種類_正答率"),
        }
        for row in result["summary"]:
            key = f'r{int(row["rep"])}_{row["T_n_ms"]:g}ms'
            for prefix in ("8素材", "3種類"):
                for suffix in ("個数", "割合"):
                    if f"{prefix}_{key}_{suffix}" not in rows_by_sheet:
                        return None
        return result
    except (
        OSError,
        BadZipFile,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        AttributeError,
        ET.ParseError,
    ):
        return None
