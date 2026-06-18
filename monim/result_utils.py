#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, TextIO, Tuple


class MethodSpec(NamedTuple):
    key: str
    label: str
    param: str
    net_param: Optional[str]
    output_rel: str


METHODS = {
    "fullmem": MethodSpec(
        key="fullmem",
        label="FullMem",
        param="0.0,0.0",
        net_param=None,
        output_rel="monim/final/fullmem_output",
    ),
    "monim": MethodSpec(
        key="monim",
        label="MoNIM",
        param="-1.5,1.0",
        net_param="-1.5_1.0",
        output_rel="output",
    ),
}

SUMMARY_FIELDS = [
    "day",
    "method",
    "param",
    "dstore_size",
    "memrate",
    "best_ppl",
    "lambda",
    "temp",
    "next_word_acc",
    "ppl_rows",
    "pwd_rows",
    "ppl_file",
]

GRID_FIELDS = ["method", "day", "kind", "rows", "unique", "missing", "duplicates", "file"]

AUDIT_FIELDS = [
    "day",
    "method",
    "dstore_size",
    "memrate",
    "keys",
    "vals",
    "index",
    "trained_index",
    "net_ckpt",
    "ppl_rows",
    "pwd_rows",
]

NET_REL = (
    "checkpoint/metak.l1.0.05.ngram0.hid128.nl4.bs64.drop0.2."
    "ftall.seed927.use_k10.metadim1/checkpoint_best.pt"
)

METRIC_PAIR_RE = re.compile(r"lmbda:\s*(?P<lambda>[-+0-9.]+),\s*temp:\s*(?P<temp>[-+0-9.]+)")
PPL_VALUE_RE = re.compile(
    r"lmbda:\s*(?P<lambda>[-+0-9.]+),\s*temp:\s*(?P<temp>[-+0-9.]+).*Perplexity:\s*(?P<value>[-+0-9.]+)"
)
PWD_VALUE_RE = re.compile(
    r"lmbda:\s*(?P<lambda>[-+0-9.]+),\s*temp:\s*(?P<temp>[-+0-9.]+).*accuracy:\s*(?P<value>[-+0-9.]+)"
)
PPL_PRESENT_RE = re.compile(r"Perplexity:\s*[-+0-9.]+")
PWD_PRESENT_RE = re.compile(r"next word prediction accuracy:\s*[-+0-9.]+")


def default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def method_output_dir(root: Path, method: MethodSpec) -> Path:
    return root / method.output_rel


def require_methods(methods_csv: str) -> List[MethodSpec]:
    keys = [method.strip().lower() for method in methods_csv.split(",") if method.strip()]
    unknown = [key for key in keys if key not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {','.join(unknown)}")
    return [METHODS[key] for key in keys]


def iter_days(begin: int, end: int, day: Optional[int] = None) -> List[int]:
    if day is not None:
        return [day]
    return list(range(begin, end + 1))


def expected_grid() -> Set[Tuple[float, float]]:
    return {(round(0.4 + i * 0.05, 2), round(8.5 + j * 0.5, 1)) for i in range(5) for j in range(15)}


def artifact_prefix(ckpt_name: str, day: int, param: str, data_type: str = "") -> str:
    return f"{ckpt_name}_{data_type}{day}__fp16_p_{param}"


def metric_name(ckpt_name: str, day: int, param: str) -> str:
    return f"{artifact_prefix(ckpt_name, day, param)}.txt"


def metric_file(output_dir: Path, ckpt_name: str, day: int, param: str, kind: str) -> Path:
    return output_dir / "debug" / kind / metric_name(ckpt_name, day, param)


def load_size_days(size_path: Path, dataset: str, ckpt_name: str) -> dict:
    sizes = json.loads(size_path.read_text())
    return sizes[dataset][ckpt_name]


def parse_metric_values(path: Path, pattern: re.Pattern) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "lambda": float(match.group("lambda")),
                    "temp": float(match.group("temp")),
                    "value": float(match.group("value")),
                    "line": line,
                }
            )
    return rows


def read_metric_pairs(path: Path) -> List[Tuple[float, float]]:
    pairs = []
    if not path.exists():
        return pairs
    for line in path.read_text(errors="replace").splitlines():
        match = METRIC_PAIR_RE.search(line)
        if match:
            pairs.append((round(float(match.group("lambda")), 2), round(float(match.group("temp")), 1)))
    return pairs


def metric_rows(path: Path, pattern: re.Pattern) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(errors="replace").splitlines() if pattern.search(line))


def has_complete_value_grid(rows: List[Dict[str, Any]], expected: Set[Tuple[float, float]]) -> bool:
    pairs = {(round(row["lambda"], 2), round(row["temp"], 1)) for row in rows}
    return len(rows) == len(expected) and pairs == expected


def best_by_ppl(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return min(rows, key=lambda row: row["value"])


def matching_pwd(pwd_rows: List[Dict[str, Any]], ppl_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pwd_rows or not ppl_row:
        return None
    for row in pwd_rows:
        if math.isclose(row["lambda"], ppl_row["lambda"]) and math.isclose(row["temp"], ppl_row["temp"]):
            return row
    return None


def summarize_rows(
    output_dir: Path,
    methods: List[MethodSpec],
    begin: int,
    end: int,
    ckpt_name: str,
    dataset: str,
    day: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    size_path = output_dir / "size.json"
    day_nodes = load_size_days(size_path, dataset, ckpt_name)
    expected = expected_grid()
    rows = []
    problems = []

    for current_day in iter_days(begin, end, day):
        day_node = day_nodes.get(str(current_day), {})
        full_size = day_node.get(METHODS["fullmem"].param)
        if not full_size:
            raise SystemExit(f"Missing FullMem size for day {current_day} in {size_path}")

        for method in methods:
            dstore_size = day_node.get(method.param)
            memrate = (dstore_size / full_size) if dstore_size else None
            ppl_file = metric_file(output_dir, ckpt_name, current_day, method.param, "ppl")
            pwd_file = metric_file(output_dir, ckpt_name, current_day, method.param, "pwd")
            ppl_rows = parse_metric_values(ppl_file, PPL_VALUE_RE)
            pwd_rows = parse_metric_values(pwd_file, PWD_VALUE_RE)
            best = best_by_ppl(ppl_rows)
            pwd = matching_pwd(pwd_rows, best)
            if (
                not best
                or not has_complete_value_grid(ppl_rows, expected)
                or not has_complete_value_grid(pwd_rows, expected)
                or not pwd
            ):
                problems.append(f"{method.label} day {current_day}: {ppl_file} / {pwd_file}")

            rows.append(
                {
                    "day": current_day,
                    "method": method.label,
                    "param": method.param,
                    "dstore_size": dstore_size if dstore_size is not None else "",
                    "memrate": f"{memrate:.6f}" if memrate is not None else "",
                    "best_ppl": f"{best['value']:.4f}" if best else "",
                    "lambda": f"{best['lambda']:.2f}" if best else "",
                    "temp": f"{best['temp']:.2f}" if best else "",
                    "next_word_acc": f"{pwd['value']:.4f}" if pwd else "",
                    "ppl_rows": len(ppl_rows),
                    "pwd_rows": len(pwd_rows),
                    "ppl_file": ppl_file,
                }
            )
    return rows, problems


def metric_grid_rows(
    output_dir: Path,
    methods: List[MethodSpec],
    begin: int,
    end: int,
    ckpt_name: str,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, ...]]]:
    expected = expected_grid()
    rows = []
    problems = []
    for method in methods:
        for day in range(begin, end + 1):
            for kind in ("ppl", "pwd"):
                path = metric_file(output_dir, ckpt_name, day, method.param, kind)
                pairs = read_metric_pairs(path)
                unique = set(pairs)
                missing = sorted(expected - unique)
                duplicates = len(pairs) - len(unique)
                rows.append(
                    {
                        "method": method.label,
                        "day": day,
                        "kind": kind,
                        "rows": len(pairs),
                        "unique": len(unique),
                        "missing": len(missing),
                        "duplicates": duplicates,
                        "file": path,
                    }
                )
                if len(pairs) != len(expected) or len(unique) != len(expected) or duplicates:
                    problems.append((method.label, day, kind, missing[:5], duplicates, path))
    return rows, problems


def artifact_audit_rows(root: Path, begin: int, end: int, ckpt_name: str, dataset: str) -> List[Dict[str, Any]]:
    size_days = load_size_days(root / "output" / "size.json", dataset, ckpt_name)
    rows = []

    for day in range(begin, end + 1):
        day_node = size_days[str(day)]
        full_size = day_node[METHODS["fullmem"].param]
        for method in METHODS.values():
            dstore_size = day_node.get(method.param)
            memrate = dstore_size / full_size if dstore_size else None
            prefix = artifact_prefix(ckpt_name, day, method.param)
            dstore = root / "storage" / ckpt_name / dataset / "dstore" / prefix
            index = root / "storage" / ckpt_name / dataset / "knn" / f"{prefix}.index"
            net_ckpt = None
            if method.net_param is not None:
                net_ckpt = (
                    root
                    / "datasets"
                    / dataset
                    / str(day)
                    / "net"
                    / "total"
                    / ckpt_name
                    / method.net_param
                    / NET_REL
                )
            metric = metric_name(ckpt_name, day, method.param)
            output_dir = method_output_dir(root, method)
            rows.append(
                {
                    "day": day,
                    "method": method.label,
                    "dstore_size": dstore_size if dstore_size is not None else "",
                    "memrate": f"{memrate:.6f}" if memrate is not None else "",
                    "keys": int((dstore.with_name(dstore.name + "_keys.npy")).exists()),
                    "vals": int((dstore.with_name(dstore.name + "_vals.npy")).exists()),
                    "index": int(index.exists()),
                    "trained_index": int((index.with_name(index.name + ".trained")).exists()),
                    "net_ckpt": "" if net_ckpt is None else int(net_ckpt.exists()),
                    "ppl_rows": metric_rows(output_dir / "debug" / "ppl" / metric, PPL_PRESENT_RE),
                    "pwd_rows": metric_rows(output_dir / "debug" / "pwd" / metric, PWD_PRESENT_RE),
                }
            )
    return rows


def write_csv(file_obj: TextIO, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_csv_file(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        write_csv(handle, fieldnames, rows)


def write_csv_stdout(fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    write_csv(sys.stdout, fieldnames, rows)
