"""Generate docs/architecture.md: Mermaid diagrams + docstring reference.

Usage: python scripts/gen_docs.py
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

SRC = Path("src/macroscan")
OUT = Path("docs/architecture.md")
PKG = "macroscan"
NO_DOC = "*(no docstring)*"


@dataclass
class FuncInfo:
    module: str                   # "b_ingest.b3_store"
    file: str                     # path to the source file, relative to the working directory
    line: int                     # line of the `def`
    qualname: str                 # "Store.read_latest" or "register"
    signature: str
    summary: str                  # first docstring line, "" if missing
    calls: set[str] = field(default_factory=set)       # fids (or "module:Class" for class stubs)
    called_by: set[str] = field(default_factory=set)
    implements: set[str] = field(default_factory=set)       # base-class methods this overrides
    implemented_by: set[str] = field(default_factory=set)   # subclass methods overriding this
    node: ast.AST | None = field(default=None, repr=False)

    @property
    def fid(self) -> str:
        return f"{self.module}:{self.qualname}"


@dataclass
class ClassInfo:
    name: str
    line: int
    bases: list[str]
    summary: str
    attrs: list[str]
    methods: list[FuncInfo]


@dataclass
class ModuleInfo:
    name: str                     # "b_ingest.b3_store"
    summary: str
    imports: set[str]             # other module names within the package
    functions: list[FuncInfo]
    classes: list[ClassInfo]
    file: str
    names: dict[str, tuple[str, str]]   # local name -> (module, symbol) for project imports
    var_types: dict[str, str]           # module-level `X: <annotation>` -> annotation text


def _summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    return doc.strip().splitlines()[0] if doc else ""  # TODO: first line only, or first paragraph?


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return f"{fn.name}({ast.unparse(fn.args)})" + (f" -> {ast.unparse(fn.returns)}" if fn.returns else "")


def _func(fn, module: str, file: str, prefix: str = "") -> FuncInfo:
    return FuncInfo(module, file, fn.lineno, prefix + fn.name, _signature(fn), _summary(fn), node=fn)


def parse_module(path: Path, root: Path) -> ModuleInfo:
    """Parse one file into a ModuleInfo. A package's __init__.py is named after the package."""
    tree = ast.parse(path.read_text())
    parts = path.relative_to(root).with_suffix("").parts
    name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(PKG + "."):
            imports.add(n.module.removeprefix(PKG + "."))  # TODO: relative imports aren't handled
    names, var_types = {}, {}
    funcs, classes = [], []
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(PKG + "."):
            for a in n.names:
                names[a.asname or a.name] = (n.module.removeprefix(PKG + "."), a.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            var_types[n.target.id] = ast.unparse(n.annotation)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(_func(n, name, str(path)))
        elif isinstance(n, ast.ClassDef):
            attrs = [
                ast.unparse(s.target) + (f": {ast.unparse(s.annotation)}" if s.annotation else "")
                for s in n.body if isinstance(s, ast.AnnAssign)
            ]
            methods = [_func(m, name, str(path), n.name + ".") for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append(ClassInfo(n.name, n.lineno, [ast.unparse(b) for b in n.bases], _summary(n), attrs, methods))
    return ModuleInfo(name, _summary(tree), imports, funcs, classes, str(path), names, var_types)


def collect(root: Path) -> list[ModuleInfo]:
    """Parse every non-empty module under root, sorted by path (a1, a2, b1, ...)."""
    files = sorted(p for p in root.rglob("*.py") if p.stat().st_size > 0)
    return [parse_module(p, root) for p in files]


class Resolver:
    """Resolves names and calls to project functions without guessing."""

    def __init__(self, modules: list[ModuleInfo]):
        self.mods = {m.name: m for m in modules}
        self.classes = {(m.name, c.name): c for m in modules for c in m.classes}
        self.funcs = {
            f.fid: f for m in modules for f in m.functions + [x for c in m.classes for x in c.methods]
        }

    def lookup(self, mod: str, name: str, seen: frozenset = frozenset()) -> tuple | None:
        """('func', fid) | ('class', mod, name) | ('module', mod) | ('var', mod, name) | None."""
        if (mod, name) in seen:
            return None
        if f"{mod}:{name}" in self.funcs:
            return ("func", f"{mod}:{name}")
        if (mod, name) in self.classes:
            return ("class", mod, name)
        m = self.mods[mod]
        if name in m.var_types:
            return ("var", mod, name)
        if name in m.names:
            src, sym = m.names[name]
            if f"{src}.{sym}" in self.mods:
                return ("module", f"{src}.{sym}")
            if src in self.mods:
                return self.lookup(src, sym, seen | {(mod, name)})   # follow re-exports
        return None

    def mro(self, key: tuple[str, str]) -> list[tuple[str, str]]:
        out = [key]
        for b in self.classes[key].bases:
            r = self.lookup(key[0], b.split("[")[0])
            if r and r[0] == "class":
                out += [k for k in self.mro(r[1:]) if k not in out]
        return out

    def method(self, key: tuple[str, str], name: str) -> str | None:
        for k in self.mro(key):
            fid = f"{k[0]}:{k[1]}.{name}"
            if fid in self.funcs:
                return fid
        return None

    def _class_in(self, mod: str, annotation: str) -> tuple[str, str] | None:
        """First project class named in an annotation string, e.g. 'list[Dimension] | None'."""
        for ident in re.findall(r"[A-Za-z_]\w*", annotation):
            r = self.lookup(mod, ident)
            if r and r[0] == "class":
                return r[1:]
        return None

    def _instance_of(self, mod: str, e: ast.expr, types: dict) -> tuple[str, str] | None:
        if isinstance(e, ast.Name):
            return types.get(e.id)
        if isinstance(e, ast.BoolOp):
            return next(filter(None, (self._instance_of(mod, v, types) for v in e.values)), None)
        if isinstance(e, ast.Call):
            fn = e.func
            if isinstance(fn, ast.Name):
                r = self.lookup(mod, fn.id)
                return r[1:] if r and r[0] == "class" else None
            if isinstance(fn, ast.Subscript) and isinstance(fn.value, ast.Name):   # SOURCES[x](...)
                r = self.lookup(mod, fn.value.id)
                if r and r[0] == "var":
                    return self._class_in(r[1], self.mods[r[1]].var_types[r[2]])
        return None

    def local_types(self, f: FuncInfo) -> dict[str, tuple[str, str]]:
        """Variable name -> project class, from self/cls, parameter annotations and assignments."""
        types: dict[str, tuple[str, str]] = {}
        a = f.node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs:
            if arg.arg in ("self", "cls") and "." in f.qualname:
                types[arg.arg] = (f.module, f.qualname.split(".")[0])
            elif arg.annotation is not None:
                k = self._class_in(f.module, ast.unparse(arg.annotation))
                if k:
                    types[arg.arg] = k
        assigns = sorted((n for n in ast.walk(f.node) if isinstance(n, (ast.Assign, ast.AnnAssign))),
                         key=lambda n: n.lineno)
        for n in assigns:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and n.value is not None:
                k = self._instance_of(f.module, n.value, types)
                if k:
                    types.setdefault(targets[0].id, k)   # TODO: first assignment wins (no flow analysis)
        return types

    def _target(self, r: tuple | None) -> str | None:
        if not r:
            return None
        if r[0] == "func":
            return r[1]
        if r[0] == "class":
            return self.method(r[1:], "__init__") or f"{r[1]}:{r[2]}"
        return None

    def resolve(self, f: FuncInfo, call: ast.Call, types: dict) -> str | None:
        fn = call.func
        if isinstance(fn, ast.Name):
            return self._target(self.lookup(f.module, fn.id))
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            v = fn.value.id
            if v in types:
                return self.method(types[v], fn.attr)
            r = self.lookup(f.module, v)
            if r and r[0] == "class":
                return self.method(r[1:], fn.attr)
            if r and r[0] == "module":
                return self._target(self.lookup(r[1], fn.attr))
        return None   # stdlib / third-party / dynamic: no edge


def resolve_calls(modules: list[ModuleInfo]) -> dict[str, FuncInfo]:
    """Fill calls/called_by and override links; returns the fid -> FuncInfo index."""
    r = Resolver(modules)
    for f in r.funcs.values():
        types = r.local_types(f)
        for n in ast.walk(f.node):
            if isinstance(n, ast.Call):
                t = r.resolve(f, n, types)
                if t and t != f.fid:
                    f.calls.add(t)
                    if t in r.funcs:
                        r.funcs[t].called_by.add(f.fid)
    for key, c in r.classes.items():
        for m in c.methods:
            name = m.qualname.split(".")[-1]
            if name.startswith("__"):
                continue
            for base in r.mro(key)[1:]:
                bid = f"{base[0]}:{base[1]}.{name}"
                if bid in r.funcs:
                    m.implements.add(bid)
                    r.funcs[bid].implemented_by.add(m.fid)
                    break
    return r.funcs


def _mid(name: str) -> str:
    return name.replace(".", "_")


def _anchor(kind: str, name: str) -> str:
    """Stable HTML id for a module ('mod'), class ('cls') or function ('fn', name is a fid)."""
    if kind == "fn":
        mod, q = name.split(":")
        return f"fn-{mod.replace('.', '-')}-{q}"
    return f"{kind}-{name.replace('.', '-') if kind == 'mod' else name}"


def _nid(fid: str) -> str:
    return "n_" + re.sub(r"\W", "_", fid)


def _link_anchor(fid: str, funcs: dict[str, FuncInfo]) -> str:
    return _anchor("fn", fid) if fid in funcs else _anchor("cls", fid.split(":")[1])


def render_call_graph(m: ModuleInfo, funcs: dict[str, FuncInfo]) -> str:
    """Call graph for one module: its functions plus direct callers/callees elsewhere."""
    own = {f.fid for f in m.functions + [x for c in m.classes for x in c.methods]}
    edges: set[tuple[str, str, bool]] = set()
    for fid in own:
        f = funcs[fid]
        edges |= {(fid, t, False) for t in f.calls} | {(c, fid, False) for c in f.called_by}
        edges |= {(b, fid, True) for b in f.implements} | {(fid, d, True) for d in f.implemented_by}
    if not edges:
        return ""
    ids = {i for s, d, _ in edges for i in (s, d)}
    lines = ["flowchart LR", f'  subgraph this["{m.name}"]']
    lines += [f'    {_nid(i)}["{funcs[i].qualname}"]' for i in sorted(ids & own)]
    lines.append("  end")
    for i in sorted(ids - own):   # callers/callees in other modules, or class stubs
        mod, q = i.split(":")
        lines.append(f'  {_nid(i)}["{mod.split(".")[-1]}.{q}"]:::ext')
    for s, d, dotted in sorted(edges):
        lines.append(f"  {_nid(d)} -.->|implements| {_nid(s)}" if dotted else f"  {_nid(d)} --> {_nid(s)}")
    lines += [f'  click {_nid(i)} href "#{_link_anchor(i, funcs)}"' for i in sorted(ids)]
    lines.append("  classDef ext stroke-dasharray: 4 3")
    return "\n".join(lines)


def render_import_graph(modules: list[ModuleInfo]) -> str:
    lines = ["flowchart LR"]
    pkgs: dict[str, list[ModuleInfo]] = {}
    for m in modules:
        pkgs.setdefault(m.name.split(".")[0], []).append(m)
    for pkg, ms in pkgs.items():
        lines.append(f"  subgraph {pkg}")
        lines += [f'    {_mid(m.name)}["{m.name.split(".")[-1]}"]' for m in ms]
        lines.append("  end")
    known = {m.name for m in modules}
    for m in modules:
        lines += [f"  {_mid(i)} --> {_mid(m.name)}" for i in sorted(m.imports & known - {m.name})]
    lines += [f'  click {_mid(m.name)} href "#{_anchor("mod", m.name)}"' for m in modules]
    return "\n".join(lines)


def _mm(text: str) -> str:
    """Make a type/signature safe for Mermaid classDiagram (generics use ~, no | or ->)."""
    for old, new in (("[", "~"), ("]", "~"), (" | ", " or "), ("->", ":"), ("{", "("), ("}", ")")):
        text = text.replace(old, new)
    return text


def render_class_diagram(modules: list[ModuleInfo]) -> str:
    lines = ["classDiagram"]
    all_classes = [c for m in modules for c in m.classes]
    for c in all_classes:
        lines.append(f"  class {c.name} {{")
        lines += [f"    {_mm(a)}" for a in c.attrs]
        lines += [f"    {_mm(m.signature)}" for m in c.methods if not m.qualname.endswith(".__init__")]
        lines.append("  }")
        lines += [f"  {b} <|-- {c.name}" for b in c.bases if b not in ("ABC",)]  # TODO: hide ABC/object bases?
    lines += [f'  click {c.name} href "#{_anchor("cls", c.name)}"' for c in all_classes]
    return "\n".join(lines)


def render_module_index(modules: list[ModuleInfo]) -> str:
    """Linked table of modules and classes; works in every Markdown viewer."""
    rows = ["| Module | What it does |", "|---|---|"]
    for m in modules:
        summary = m.summary.splitlines()[0] if m.summary else ""
        rows.append(f"| [`{m.name}`](#{_anchor('mod', m.name)}) | {summary} |")
    rows += ["", "| Class | What it does |", "|---|---|"]
    for c in (c for m in modules for c in m.classes):
        rows.append(f"| [`{c.name}`](#{_anchor('cls', c.name)}) | {c.summary} |")
    return "\n".join(rows)


def _src(file: str, line: int, docs_dir: Path) -> str:
    """Relative link from the generated doc to a line of source (works on GitHub and in VS Code)."""
    return Path(os.path.relpath(file, docs_dir)).as_posix() + (f"#L{line}" if line else "")


def _links(ids: set[str], here: str, funcs: dict[str, FuncInfo]) -> str:
    def label(i: str) -> str:
        mod, q = i.split(":")
        return q if mod == here else f"{mod.split('.')[-1]}.{q}"
    return ", ".join(f"[`{label(i)}`](#{_link_anchor(i, funcs)})" for i in sorted(ids))


def _returns(f: FuncInfo) -> str:
    """Returned variable(s)/expression(s) with the annotated return type."""
    exprs: list[str] = []
    stack = list(f.node.body)
    while stack:  # walk the body without descending into nested functions/classes/lambdas
        n = stack.pop(0)
        if isinstance(n, ast.Return) and n.value is not None:
            e = ast.unparse(n.value)
            e = e if len(e) <= 40 else e[:37] + "..."
            if e != "None" and e not in exprs:
                exprs.append(e)
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack = list(ast.iter_child_nodes(n)) + stack
    rtype = ast.unparse(f.node.returns) if f.node.returns else None
    if not exprs:
        return "*(none)*" if rtype in (None, "None") else f"`{rtype}`"
    names = ", ".join(f"`{e}`" for e in exprs)
    return f"{names}: `{rtype}`" if rtype else f"{names}: *(unannotated)*"


def _entry(f: FuncInfo, funcs: dict[str, FuncInfo], docs_dir: Path) -> list[str]:
    out = [f'- <a id="{_anchor("fn", f.fid)}"></a>**[`{f.signature}`]({_src(f.file, f.line, docs_dir)})** — {f.summary or NO_DOC}']
    out.append(f"  - returns: {_returns(f)}")
    out.append(f"  - calls: {_links(f.calls, f.module, funcs) if f.calls else '*(none)*'}")
    for label, ids in (("called by", f.called_by),
                       ("overrides", f.implements), ("implemented by", f.implemented_by)):
        if ids:
            out.append(f"  - {label}: {_links(ids, f.module, funcs)}")
    return out


def render_reference(modules: list[ModuleInfo], funcs: dict[str, FuncInfo], docs_dir: Path) -> str:
    lines = []
    for m in modules:
        lines += [f'### <a id="{_anchor("mod", m.name)}"></a>[`{m.name}`]({_src(m.file, 0, docs_dir)})', "", m.summary, ""]
        graph = render_call_graph(m, funcs)
        if graph:
            lines += ["```mermaid", graph, "```", ""]
        for f in m.functions:
            lines += _entry(f, funcs, docs_dir)
        for c in m.classes:
            lines += [f'- <a id="{_anchor("cls", c.name)}"></a>**[class `{c.name}`]({_src(m.file, c.line, docs_dir)})** — {c.summary or NO_DOC}']
            for meth in c.methods:
                lines += ["  " + l for l in _entry(meth, funcs, docs_dir)]
        lines.append("")
    return "\n".join(lines)


def main(src: Path = SRC, out: Path = OUT) -> None:
    modules = collect(src)
    funcs = resolve_calls(modules)
    parts = [
        "<!-- Generated by scripts/gen_docs.py. Do not edit. -->",
        "# MacroScan architecture",
        "## Module dependencies",
        f"```mermaid\n{render_import_graph(modules)}\n```",
        render_module_index(modules),
        "## Classes",
        f"```mermaid\n{render_class_diagram(modules)}\n```",
        "## Reference",
        render_reference(modules, funcs, out.parent),
    ]
    md = "\n\n".join(parts) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
