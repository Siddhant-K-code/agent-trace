"""Eval configuration loader (.agent-evals.yaml).

Parses a YAML-like config file using stdlib only (no PyYAML dependency).
Supports a minimal subset of YAML: string keys, scalar values, and lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScorerConfig:
    type: str
    threshold: float = 1.0
    # Extra scorer-specific params stored as a dict
    params: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    scorers: list[ScorerConfig] = field(default_factory=list)
    pass_threshold: float = 0.85
    warn_threshold: float = 0.70

    @classmethod
    def default(cls) -> "EvalConfig":
        """Return a sensible default config when no file is present."""
        return cls(
            scorers=[
                ScorerConfig(type="no_errors", threshold=1.0),
            ],
            pass_threshold=0.85,
            warn_threshold=0.70,
        )


# ---------------------------------------------------------------------------
# Minimal YAML parser (stdlib only)
# ---------------------------------------------------------------------------

def _parse_yaml_value(raw: str):
    """Parse a scalar YAML value to Python type."""
    raw = raw.strip()
    if raw in ("true", "True", "yes"):
        return True
    if raw in ("false", "False", "no"):
        return False
    if raw in ("null", "~", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # Strip surrounding quotes
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _parse_minimal_yaml(text: str) -> dict:
    """Parse a minimal YAML document into a nested dict/list structure.

    Supports:
    - key: value          (scalar mapping)
    - key:                (mapping block — indented children follow)
    - - key: value        (list of mappings)
    - - scalar            (list of scalars)
    """
    lines = text.splitlines()

    def _skip_blanks(idx: int) -> int:
        while idx < len(lines) and (not lines[idx].strip() or lines[idx].strip().startswith("#")):
            idx += 1
        return idx

    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    def _parse_block(start: int, base_indent: int) -> tuple[int, object]:
        """Parse a block starting at *start* with *base_indent* indentation.

        Returns (next_line_index, parsed_value).
        """
        i = _skip_blanks(start)
        if i >= len(lines):
            return i, {}

        first_content = lines[i].strip()

        # Determine block type from first non-blank line
        if first_content.startswith("- "):
            # List block
            result_list: list = []
            while i < len(lines):
                i = _skip_blanks(i)
                if i >= len(lines):
                    break
                line = lines[i]
                ind = _indent(line)
                content = line.strip()
                if ind < base_indent:
                    break
                if not content.startswith("- "):
                    break

                item_content = content[2:].strip()
                if ":" in item_content:
                    # Mapping item: parse key: value on this line, then sub-keys
                    item_dict: dict = {}
                    key, _, rest = item_content.partition(":")
                    key = key.strip()
                    rest = rest.strip()
                    if rest:
                        item_dict[key] = _parse_yaml_value(rest)
                    else:
                        # value is a sub-block
                        i += 1
                        i, sub = _parse_block(i, ind + 2)
                        item_dict[key] = sub
                        result_list.append(item_dict)
                        continue

                    # Collect additional key: value lines at deeper indent
                    i += 1
                    while i < len(lines):
                        i = _skip_blanks(i)
                        if i >= len(lines):
                            break
                        sub_line = lines[i]
                        sub_ind = _indent(sub_line)
                        sub_content = sub_line.strip()
                        if sub_ind <= ind or sub_content.startswith("- "):
                            break
                        if ":" in sub_content:
                            k, _, v = sub_content.partition(":")
                            item_dict[k.strip()] = _parse_yaml_value(v.strip())
                        i += 1
                    result_list.append(item_dict)
                else:
                    result_list.append(_parse_yaml_value(item_content))
                    i += 1

            return i, result_list
        else:
            # Mapping block
            result_dict: dict = {}
            while i < len(lines):
                i = _skip_blanks(i)
                if i >= len(lines):
                    break
                line = lines[i]
                ind = _indent(line)
                content = line.strip()
                if ind < base_indent:
                    break
                if content.startswith("- "):
                    break
                if ":" not in content:
                    i += 1
                    continue

                key, _, rest = content.partition(":")
                key = key.strip()
                rest = rest.strip()
                i += 1

                if rest:
                    result_dict[key] = _parse_yaml_value(rest)
                else:
                    # Look ahead for child block
                    j = _skip_blanks(i)
                    if j < len(lines) and _indent(lines[j]) > ind:
                        i, child = _parse_block(i, _indent(lines[j]))
                        result_dict[key] = child
                    else:
                        result_dict[key] = None

            return i, result_dict

    _, result = _parse_block(0, 0)
    if not isinstance(result, dict):
        return {}
    return result


def load_config(path: str | Path = ".agent-evals.yaml") -> EvalConfig:
    """Load eval config from *path*. Returns default config if file not found."""
    p = Path(path)
    if not p.exists():
        return EvalConfig.default()

    try:
        text = p.read_text(encoding="utf-8")
        data = _parse_minimal_yaml(text)
    except Exception:
        return EvalConfig.default()

    scorers: list[ScorerConfig] = []
    for s in data.get("scorers", []):
        if not isinstance(s, dict):
            continue
        scorer_type = str(s.get("type", ""))
        if not scorer_type:
            continue
        threshold = float(s.get("threshold", s.get("weight", 1.0)))
        params = {k: v for k, v in s.items() if k not in ("type", "threshold", "weight")}
        scorers.append(ScorerConfig(type=scorer_type, threshold=threshold, params=params))

    thresholds = data.get("thresholds", {}) or {}
    pass_t = float(thresholds.get("pass", 0.85)) if isinstance(thresholds, dict) else 0.85
    warn_t = float(thresholds.get("warn", 0.70)) if isinstance(thresholds, dict) else 0.70

    if not scorers:
        scorers = EvalConfig.default().scorers

    return EvalConfig(scorers=scorers, pass_threshold=pass_t, warn_threshold=warn_t)
