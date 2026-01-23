import os
import ast
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import streamlit as st


# =========================
# Data structures
# =========================

@dataclass
class DocTarget:
    name: str
    node_type: str
    lineno: int
    col_offset: int
    has_docstring: bool
    current_docstring: Optional[str]
    suggested_docstring: Optional[str]


@dataclass
class CoverageReport:
    total_targets: int
    documented_targets: int
    undocumented_targets: int
    documentation_percentage: float
    pydocstyle_passed: bool
    pydocstyle_output: str


# =========================
# AST helpers: Raises/Yields/Attributes
# =========================

def infer_raises(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and n.exc is not None:
            if isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name):
                names.add(n.exc.func.id)
            elif isinstance(n.exc, ast.Name):
                names.add(n.exc.id)
    return names


def infer_yields(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def infer_class_attributes(node: ast.ClassDef) -> List[str]:
    attrs: List[str] = []
    for n in node.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    attrs.append(target.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            attrs.append(n.target.id)
    return attrs


def has_return_value(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Return) and n.value is not None:
            return True
    return False


# =========================
# Style-specific builders
# =========================

def build_google_docstring(
    name: str,
    args: List[str],
    has_ret: bool,
    raises: Set[str],
    is_generator: bool,
    attributes: Optional[List[str]] = None,
) -> str:
    """Build Google style docstring.[web:16]"""
    lines: List[str] = []
    lines.append(f"{name.replace('_', ' ').capitalize()}.")
    lines.append("")

    if args:
        lines.append("Args:")
        for a in args:
            lines.append(f"    {a}: Description of {a}.")
        lines.append("")

    if attributes:
        lines.append("Attributes:")
        for attr in attributes:
            lines.append(f"    {attr}: Description of {attr}.")
        lines.append("")

    if is_generator:
        lines.append("Yields:")
        lines.append("    Any: Description of yielded values.")
        lines.append("")
    elif has_ret:
        lines.append("Returns:")
        lines.append("    Any: Description of return value.")
        lines.append("")

    if raises:
        lines.append("Raises:")
        for r in raises:
            lines.append(f"    {r}: Description of when {r} is raised.")
        lines.append("")

    return '"""' + "\n".join(lines).rstrip() + '\n"""'


def build_numpy_docstring(
    name: str,
    args: List[str],
    has_ret: bool,
    raises: Set[str],
    is_generator: bool,
    attributes: Optional[List[str]] = None,
) -> str:
    """Build NumPy style docstring.[web:30][web:35]"""
    lines: List[str] = []
    lines.append(f"{name.replace('_', ' ').capitalize()}.")
    lines.append("")

    if args:
        lines.append("Parameters")
        lines.append("----------")
        for a in args:
            lines.append(f"{a} : Any")
            lines.append(f"    Description of {a}.")
        lines.append("")

    if attributes:
        lines.append("Attributes")
        lines.append("----------")
        for attr in attributes:
            lines.append(f"{attr} : Any")
            lines.append(f"    Description of {attr}.")
        lines.append("")

    if is_generator:
        lines.append("Yields")
        lines.append("------")
        lines.append("Any")
        lines.append("    Description of yielded values.")
        lines.append("")
    elif has_ret:
        lines.append("Returns")
        lines.append("-------")
        lines.append("Any")
        lines.append("    Description of return value.")
        lines.append("")

    if raises:
        lines.append("Raises")
        lines.append("------")
        for r in raises:
            lines.append(f"{r}")
            lines.append(f"    Description of when {r} is raised.")
        lines.append("")

    return '"""' + "\n".join(lines).rstrip() + '\n"""'


def build_rest_docstring(
    name: str,
    args: List[str],
    has_ret: bool,
    raises: Set[str],
    is_generator: bool,
    attributes: Optional[List[str]] = None,
) -> str:
    """Build reST style docstring using field lists.[web:25][web:31]"""
    lines: List[str] = []
    lines.append(f"{name.replace('_', ' ').capitalize()}.")
    lines.append("")

    for a in args:
        lines.append(f":param {a}: Description of {a}.")
    if attributes:
        for attr in attributes:
            lines.append(f":ivar {attr}: Description of {attr}.")
    if is_generator:
        lines.append(":yields: Description of yielded values.")
    elif has_ret:
        lines.append(":returns: Description of return value.")
    for r in raises:
        lines.append(f":raises {r}: Description of when {r} is raised.")

    return '"""' + "\n".join(lines).rstrip() + '\n"""'


def build_docstring_for_node(node: ast.AST, style: str) -> str:
    """Build docstring for a node in given style.[web:24]"""
    name = getattr(node, "name", "object")
    args: List[str] = []
    attributes: Optional[List[str]] = None
    raises = infer_raises(node)
    is_gen = infer_yields(node)
    has_ret = has_return_value(node)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args if a.arg != "self"]
    elif isinstance(node, ast.ClassDef):
        attributes = infer_class_attributes(node)

    style_lower = style.lower()
    if style_lower == "google":
        return build_google_docstring(name, args, has_ret, raises, is_gen, attributes)
    if style_lower == "numpy":
        return build_numpy_docstring(name, args, has_ret, raises, is_gen, attributes)
    if style_lower in {"rest", "restructuredtext", "restructured"}:
        return build_rest_docstring(name, args, has_ret, raises, is_gen, attributes)

    # fallback
    return build_google_docstring(name, args, has_ret, raises, is_gen, attributes)


# =========================
# Analyzer and code modifier
# =========================

class DocstringAnalyzer(ast.NodeVisitor):
    """Analyze functions/classes and collect docstring information."""

    def __init__(self, style: str) -> None:
        self.targets: List[DocTarget] = []
        self.style = style

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_target(node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_target(node, "async function")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_target(node, "class")
        self.generic_visit(node)

    def _add_target(self, node: ast.AST, node_type: str) -> None:
        doc = ast.get_docstring(node)
        has_doc = doc is not None
        suggested = None
        if not has_doc:
            suggested = build_docstring_for_node(node, self.style)

        self.targets.append(
            DocTarget(
                name=getattr(node, "name", "<anonymous>"),
                node_type=node_type,
                lineno=getattr(node, "lineno", -1),
                col_offset=getattr(node, "col_offset", 0),
                has_docstring=has_doc,
                current_docstring=doc,
                suggested_docstring=suggested,
            )
        )


def analyze_source(code: str, style: str) -> List[DocTarget]:
    tree = ast.parse(code)
    analyzer = DocstringAnalyzer(style)
    analyzer.visit(tree)
    return analyzer.targets


def apply_generated_docstrings(code: str, targets: List[DocTarget]) -> str:
    """
    Insert generated docstrings into the original source code.

    Strategy:
    - For each undocumented node, insert suggested docstring right after
      the function/class definition line with proper indentation.[web:11]
    """
    if not targets:
        return code

    lines = code.splitlines()
    missing = [t for t in targets if not t.has_docstring and t.suggested_docstring]
    missing_sorted = sorted(missing, key=lambda t: t.lineno, reverse=True)

    for t in missing_sorted:
        idx = t.lineno - 1
        if idx < 0 or idx >= len(lines):
            continue

        def_line = lines[idx]
        indent = " " * (len(def_line) - len(def_line.lstrip(" ")))
        doc_lines = [indent + l for l in t.suggested_docstring.splitlines()]

        insertion_index = idx + 1
        if insertion_index < len(lines) and lines[insertion_index].strip():
            lines.insert(insertion_index, indent + "")
        insertion_index = idx + 1
        for offset, dl in enumerate(doc_lines):
            lines.insert(insertion_index + offset, dl)

    return "\n".join(lines)


# =========================
# pydocstyle integration
# =========================

def run_pydocstyle_on_code(code: str) -> Tuple[bool, str]:
    """Run pydocstyle on a temporary file containing code.[web:8][web:11]"""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "temp_code.py")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            proc = subprocess.run(
                ["pydocstyle", temp_path],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return False, "pydocstyle command not found. Please ensure it is installed and on PATH."

        output = proc.stdout + proc.stderr
        passed = proc.returncode == 0
        return passed, output.strip()


def create_coverage_report(targets: List[DocTarget], pydocstyle_result: Tuple[bool, str]) -> CoverageReport:
    total = len(targets)
    documented = sum(1 for t in targets if t.has_docstring)
    undocumented = total - documented
    percentage = (documented / total * 100.0) if total > 0 else 0.0
    passed, output_text = pydocstyle_result

    return CoverageReport(
        total_targets=total,
        documented_targets=documented,
        undocumented_targets=undocumented,
        documentation_percentage=percentage,
        pydocstyle_passed=passed,
        pydocstyle_output=output_text,
    )


# =========================
# Streamlit UI helpers
# =========================

def set_page_config() -> None:
    st.set_page_config(
        page_title="PEP 257 Docstring Studio",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #111827, #020617 55%);
            color: #e5e7eb;
        }
        .glass-panel {
            background: rgba(15, 23, 42, 0.80);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.8);
        }
        .metric-pill {
            background: linear-gradient(135deg, #1d4ed8, #7c3aed);
            border-radius: 999px;
            padding: 0.6rem 1.1rem;
            color: #f9fafb;
            font-weight: 600;
            font-size: 0.9rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .metric-pill span.value {
            font-size: 1.1rem;
        }
        .code-container {
            border-radius: 14px;
            background: radial-gradient(circle at top, #0b1120, #020617);
            border: 1px solid rgba(55, 65, 81, 0.9);
        }
        .status-badge-pass {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.8rem;
            border: 1px solid rgba(34, 197, 94, 0.4);
        }
        .status-badge-fail {
            background: rgba(248, 113, 113, 0.15);
            color: #fecaca;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.8rem;
            border: 1px solid rgba(248, 113, 113, 0.4);
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #9ca3af;
        }
        .highlight-gradient {
            background: linear-gradient(120deg, #38bdf8, #a855f7, #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .big-title {
            font-size: 2.6rem;
            font-weight: 700;
        }
        .subtle {
            color: #9ca3af;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_content() -> Dict[str, object]:
    st.sidebar.markdown("### ⚙️ Controls")

    analysis_mode = st.sidebar.radio(
        "Mode",
        ["Analyze only", "Generate + Analyze"],
        index=1,
    )

    style = st.sidebar.selectbox(
        "Docstring style",
        ["Google", "NumPy", "reST"],
        index=0,
    )

    show_raw_pydocstyle = st.sidebar.checkbox(
        "Show raw pydocstyle output",
        value=False,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PEP 257 essentials**")
    st.sidebar.markdown(
        "- Triple double quotes.\n"
        "- One-line summary first.\n"
        "- Blank line before sections.\n"
        "- Imperative mood for summaries.[web:5]"
    )

    return {
        "generate": analysis_mode == "Generate + Analyze",
        "show_raw_pydocstyle": show_raw_pydocstyle,
        "style": style,
    }


def render_header() -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Milestone 2 · PEP 257 Enforcement</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="big-title"><span class="highlight-gradient">Docstring Studio</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="subtle">Generate Google, NumPy, or reST docstrings with Raises/Yields/Attributes and validate them with pydocstyle.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("**Purpose**")
        st.markdown(
            "- Enforce PEP 257\n"
            "- Detect formatting issues\n"
            "- Coverage & compliance reporting",
        )
        st.markdown("**Styles**")
        st.markdown(
            "- Google style\n"
            "- NumPy style\n"
            "- reStructuredText (reST)",
        )
        st.markdown("</div>", unsafe_allow_html=True)


def display_metrics(report: CoverageReport) -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-pill">Total <span class="value">{report.total_targets}</span></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-pill">Documented <span class="value">{report.documented_targets}</span></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-pill">Undoc <span class="value">{report.undocumented_targets}</span></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-pill">Coverage <span class="value">{report.documentation_percentage:.1f}%</span></div>',
            unsafe_allow_html=True,
        )


def render_targets_table(targets: List[DocTarget]) -> None:
    if not targets:
        st.info("No functions, classes, or async functions found.")
        return

    rows = []
    for t in targets:
        status = "✅ Documented" if t.has_docstring else "⚠️ Missing"
        rows.append(
            {
                "Name": t.name,
                "Type": t.node_type,
                "Line": t.lineno,
                "Status": status,
            }
        )

    st.table(rows)


# =========================
# Streamlit main
# =========================

def main() -> None:
    set_page_config()
    inject_custom_css()
    controls = sidebar_content()
    render_header()

    st.markdown("### 1. Input source code")
    st.markdown(
        "Upload a `.py` file or paste Python code below. "
        "The tool will analyze functions, async functions, and classes."
    )

    upload_col, paste_col = st.columns(2)
    code_text = ""
    filename_label: Optional[str] = None

    with upload_col:
        uploaded = st.file_uploader("Upload Python file", type=["py"])
        if uploaded:
            code_text = uploaded.read().decode("utf-8")
            filename_label = uploaded.name

    with paste_col:
        default_snippet = """\
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("b cannot be zero")
    return a / b


class Calculator:
    factor = 10

    def multiply(self, x, y):
        return x * y * self.factor
"""
        text_input = st.text_area(
            "Or paste Python code",
            value=code_text or default_snippet,
            height=260,
        )
        if text_input.strip():
            code_text = text_input
            filename_label = filename_label or "<pasted code>"

    if not code_text.strip():
        st.warning("Please provide Python source code to continue.")
        return

    st.markdown("### 2. Analysis & generation")

    if st.button("Run analysis", type="primary"):
        with st.spinner("Analyzing docstrings and running PEP 257 checks..."):
            targets = analyze_source(code_text, controls["style"])
            working_code = code_text

            if controls["generate"]:
                working_code = apply_generated_docstrings(working_code, targets)
                targets = analyze_source(working_code, controls["style"])

            pydocstyle_result = run_pydocstyle_on_code(working_code)
            report = create_coverage_report(targets, pydocstyle_result)

        st.markdown("#### Coverage & compliance overview")
        display_metrics(report)

        status_class = "status-badge-pass" if report.pydocstyle_passed else "status-badge-fail"
        status_text = "pydocstyle compliant" if report.pydocstyle_passed else "pydocstyle violations found"
        st.markdown(
            f'<span class="{status_class}">PEP 257 status: {status_text}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Documentation targets")
        render_targets_table(targets)

        st.markdown("#### Generated / updated source")
        st.markdown('<div class="glass-panel code-container">', unsafe_allow_html=True)
        st.code(working_code, language="python")
        st.markdown("</div>", unsafe_allow_html=True)

        if controls["show_raw_pydocstyle"]:
            st.markdown("#### Raw pydocstyle output")
            if not report.pydocstyle_output:
                st.info("No pydocstyle messages. Your file is fully compliant.")
            else:
                st.code(report.pydocstyle_output, language="text")

        st.download_button(
            "Download updated file",
            data=working_code,
            file_name=os.path.basename(filename_label or "updated_code.py"),
            mime="text/x-python",
        )


if __name__ == "__main__":
    main()
