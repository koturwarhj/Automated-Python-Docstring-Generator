import ast
from typing import Dict, Any, List, Tuple

import streamlit as st  # pip install streamlit


# ---------- ANALYSIS HELPERS ----------

def analyze_source(source: str) -> Dict[str, Any]:
    """Parse source and collect modules, classes, functions, methods, coverage."""
    tree = ast.parse(source)

    modules = ["__main__"]  # single uploaded module
    classes: List[ast.ClassDef] = []
    functions: List[ast.FunctionDef] = []
    methods: List[Tuple[str, ast.FunctionDef]] = []  # (class_name, method_node)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node)
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append((node.name, child))
        elif isinstance(node, ast.FunctionDef):
            # exclude methods here; keep only top-level functions
            if not isinstance(getattr(node, "parent", None), ast.ClassDef):
                functions.append(node)

    # attach parent info to distinguish methods vs functions
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)

    all_funcs = functions + [m for _, m in methods]
    total = len(all_funcs)
    with_doc = sum(1 for f in all_funcs if ast.get_docstring(f) is not None)
    without_doc = total - with_doc
    percent = 0.0 if total == 0 else round(with_doc * 100 / total, 1)

    return {
        "tree": tree,
        "modules": modules,
        "classes": classes,
        "functions": functions,
        "methods": methods,
        "total": total,
        "with_doc": with_doc,
        "without_doc": without_doc,
        "percent": percent,
    }


def function_signature_info(func: ast.FunctionDef) -> Dict[str, Any]:
    """Extract name, params, defaults, type hints, return."""
    args = func.args
    arg_names = [a.arg for a in args.args]

    # defaults align to last N positional args
    defaults = args.defaults
    defaults_map = {}
    if defaults:
        for name, default_node in zip(arg_names[-len(defaults):], defaults):
            defaults_map[name] = ast.unparse(default_node)

    annotations = {}
    for a in args.args:
        if a.annotation is not None:
            annotations[a.arg] = ast.unparse(a.annotation)

    return_annotation = ast.unparse(func.returns) if func.returns else None

    return {
        "name": func.name,
        "params": arg_names,
        "defaults": defaults_map,
        "annotations": annotations,
        "return_type": return_annotation,
    }


def generate_baseline_docstring(info: Dict[str, Any]) -> str:
    """Generate a baseline Google-style docstring."""
    lines = [f'"""TODO: Describe {info["name"]}."""', ""]
    params = [p for p in info["params"] if p != "self"]

    if params:
        lines.append("Args:")
        for p in params:
            default = f" (default={info['defaults'][p]})" if p in info["defaults"] else ""
            annotation = f" ({info['annotations'][p]})" if p in info["annotations"] else ""
            lines.append(f"    {p}{annotation}{default}: TODO: describe.")
        lines.append("")

    if info["return_type"] and info["return_type"] != "None":
        lines.append("Returns:")
        lines.append(f"    {info['return_type']}: TODO: describe.")
        lines.append("")

    lines.append('"""')
    return "\n".join(lines)


# ---------- STREAMLIT UI (STYLED) ----------

st.set_page_config(
    page_title="Docstring Coverage Studio",
    layout="wide",
    page_icon="🧾",
)

st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left,#111827,#020617 55%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1, h2, h3, h4 {
        color: #e5e7eb !important;
    }
    .metric-label > div {
        color: #cbd5f5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='font-size: 32px;'>Milestone 1 • Parsing & Baseline Docstring Generation</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload a Python file. The app parses it with AST, identifies modules/classes/functions/methods, "
    "extracts signatures, generates baseline docstrings, and reports docstring coverage."
)

uploaded = st.file_uploader("Drop a .py file here", type="py")

if uploaded is None:
    st.info("Upload a Python file to begin the analysis.")
else:
    source = uploaded.read().decode("utf-8")
    result = analyze_source(source)

    # ---------- Coverage metrics ----------
    col_metrics, col_progress = st.columns([1, 1.2])

    with col_metrics:
        st.markdown("### Initial docstring coverage report")
        st.metric(
            label="Coverage",
            value=f"{result['percent']}%",
            delta=f"{result['with_doc']}/{result['total']} documented",
        )
        st.metric(
            label="Functions parsed correctly",
            value=str(result["total"]),
        )

    with col_progress:
        st.markdown("### Coverage progress")
        st.progress(result["percent"] / 100.0)

    st.markdown("---")

    # ---------- AST summary: modules / classes / functions ----------
    col_left, col_right = st.columns([1.0, 1.4])

    with col_left:
        st.markdown("#### AST Structure Overview")

        st.write("**Modules**")
        st.code("\n".join(result["modules"]) or "None", language="text")

        st.write("**Classes**")
        class_names = [c.name for c in result["classes"]]
        st.code("\n".join(class_names) or "None", language="text")

        st.write("**Top‑level functions**")
        func_names = [f.name for f in result["functions"]]
        st.code("\n".join(func_names) or "None", language="text")

        st.write("**Methods**")
        method_names = [f"{cls}.{m.name}" for cls, m in result["methods"]]
        st.code("\n".join(method_names) or "None", language="text")

    # ---------- AST parsing snapshot + baseline docstrings ----------
    with col_right:
        st.markdown("#### AST Parsing Snapshot & Baseline Docstrings")

        if result["total"] == 0:
            st.write("No functions or methods found in this file.")
        else:
            for cls_name, func_node in [(None, f) for f in result["functions"]] + result["methods"]:
                info = function_signature_info(func_node)
                title = info["name"] if cls_name is None else f"{cls_name}.{info['name']}"
                st.markdown(f"**{title}**")

                # Show original source for this function/method
                snippet = ast.get_source_segment(source, func_node) or ""
                st.code(snippet.strip(), language="python")

                # Show generated baseline docstring
                st.markdown("_Generated baseline docstring:_")
                st.code(generate_baseline_docstring(info), language="python")

