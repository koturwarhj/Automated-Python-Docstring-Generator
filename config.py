from pathlib import Path
import tomllib  # Python 3.11+; for 3.10 use 'tomli' package instead

def load_docstring_config():
    pyproject_path = Path("pyproject.toml")
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    # OR:
    # with pyproject_path.open("rb") as f:
    #     data = tomllib.load(f)

    tool_cfg = data.get("tool", {}).get("docstring_enforcer", {})

    min_coverage = tool_cfg.get("min_coverage", 80)
    ignore_errors = tool_cfg.get("ignore_errors", [])
    paths = tool_cfg.get("paths", ["src"])

    return min_coverage, ignore_errors, paths
