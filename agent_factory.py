# tools_and_agents.py
import os
import re
from pathlib import Path
from typing import Optional, List, Type, ClassVar, Set, Any

from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

# =========================================================
# Core file-coverage tools: enumerate files + read chunked
# =========================================================

class EnumArgs(BaseModel):
    directory: str = Field(..., description="Repository root directory path")
    include_exts: Optional[List[str]] = Field(
        default=[
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".json", ".md", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
            ".env", ".sh", ".bash"
        ],
        description="List of file extensions to include"
    )
    exclude_dirs: Optional[List[str]] = Field(
        default=[".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".idea", ".vscode", "dist", "build"],
        description="Directory names to exclude"
    )
    max_files: int = Field(5000, description="Hard cap on files enumerated")
    model_config = ConfigDict(extra="ignore")


class EnumerateFilesTool(BaseTool):
    name: str = "Enumerate Repository Files"
    description: str = (
        "Enumerates source files under a repository. "
        "Returns a newline-separated list of absolute file paths."
    )
    args_schema: Type[BaseModel] = EnumArgs  # Pydantic v2 requires annotation

    def _run(self, directory: str, include_exts: Optional[List[str]] = None,
             exclude_dirs: Optional[List[str]] = None, max_files: int = 5000) -> str:
        root = Path(directory).resolve()
        if not root.exists():
            return f"ERROR: Directory does not exist: {root.as_posix()}"
        include_exts = [e.lower() for e in (include_exts or [])]
        exclude_set = set(exclude_dirs or [])
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_set]
            for fname in filenames:
                if len(out) >= max_files:
                    break
                p = Path(dirpath) / fname
                if include_exts and p.suffix.lower() not in include_exts:
                    continue
                out.append(p.as_posix())
            if len(out) >= max_files:
                break
        if not out:
            return f"INFO: No files found under {root.as_posix()} with given filters."
        return "\n".join(out[:max_files])


class ReadArgs(BaseModel):
    path: str = Field(..., description="Absolute file path to read")
    page: int = Field(1, description="1-based page number")
    page_chars: int = Field(8000, description="Max characters per page")
    model_config = ConfigDict(extra="ignore")


class ReadFileChunkedTool(BaseTool):
    name: str = "Read File (Chunked)"
    description: str = (
        "Reads a file in fixed-size character pages so the agent can iterate. "
        "Returns: metadata header + the requested page + pagination info."
    )
    args_schema: Type[BaseModel] = ReadArgs  # Pydantic v2 requires annotation

    def _run(self, path: str, page: int = 1, page_chars: int = 8000) -> str:
        p = Path(path)
        if not p.exists():
            return f"ERROR: File not found: {path}"
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"ERROR: Unable to read {path}: {e}"

        if page_chars <= 0:
            page_chars = 8000
        total_pages = max(1, (len(text) + page_chars - 1) // page_chars)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_chars
        end = min(len(text), start + page_chars)
        chunk = text[start:end]

        header = (
            f"FILE: {p.as_posix()}\n"
            f"SIZE_CHARS: {len(text)}\n"
            f"PAGE: {page}/{total_pages}\n"
            f"EXT: {p.suffix or '(none)'}\n"
            "---- PAGE CONTENT START ----\n"
        )
        footer = "\n---- PAGE CONTENT END ----\n" \
                 "If more pages are available, call the tool again with page=PAGE+1."
        return header + chunk + footer


# =========================
# Dependency Analysis Tool
# =========================

class DepArgs(BaseModel):
    directory: str = Field(..., description="Path to repository root to scan.")
    file: Optional[str] = Field(
        default=None,
        description="Optional file name (e.g., requirements.txt, package.json, pyproject.toml)"
    )
    model_config = ConfigDict(extra="ignore")


class DependencyAnalysisTool(BaseTool):
    name: str = "Dependency Analysis Tool"
    description: str = (
        "Parses common dependency files (requirements.txt, package.json, pyproject.toml, etc.) "
        "and returns their contents for analysis."
    )
    args_schema: Type[BaseModel] = DepArgs  # Pydantic v2 requires annotation

    def _run(self, directory: str, file: Optional[str] = None) -> str:
        root = Path(directory).resolve()
        if not root.exists():
            return f"Directory does not exist: {root.as_posix()}"

        dependency_files = {
            "requirements.txt": "Python (pip)",
            "requirements.in": "Python (pip/constraints)",
            "constraints.txt": "Python (pip constraints)",
            "pyproject.toml": "Python (poetry/pdm)",
            "Pipfile": "Python (pipenv)",
            "package.json": "Node.js (npm)",
            "package-lock.json": "Node.js (npm lockfile)",
        }
        IGNORE_DIRS = {'.git', '.hg', '.svn', '__pycache__', 'node_modules', '.venv', 'venv'}

        def read_text(p: Path):
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None

        # Case A: a specific dependency file is requested
        if file:
            target = file.lower()
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
                for fname in filenames:
                    if fname.lower() == target:
                        p = Path(dirpath) / fname
                        eco = dependency_files.get(fname, "Unknown ecosystem")
                        content = read_text(p)
                        if content:
                            rel = p.relative_to(root)
                            return f"--- {eco} ({rel}) ---\n{content}\n"
                        return f"Found {fname} at {p.as_posix()} but could not read it."
            return f"{file} not found under {root.as_posix()}."

        # Case B: scan for all known dependency files
        found: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for fname in filenames:
                if fname.lower() in dependency_files:
                    p = Path(dirpath) / fname
                    eco = dependency_files.get(fname, "Unknown ecosystem")
                    content = read_text(p)
                    if content:
                        rel = p.relative_to(root)
                        found.append(f"--- {eco} ({rel}) ---\n{content}\n")

        if not found:
            return f"No common dependency files found under {root.as_posix()}."
        return "\n".join(found)


# =========================
# Manual Security Review Tool
# =========================

class SecScanArgs(BaseModel):
    directory: str = Field(..., description="Repo root directory to scan")
    max_files: int = Field(5000, description="Safety cap on number of files to scan")
    model_config = ConfigDict(extra="ignore")


class ManualCodeReviewTool(BaseTool):
    name: str = "Manual Code Review (for hardcoded secrets)"
    description: str = (
        "Greps source files for common secret patterns (keys, tokens, passwords) "
        "and simple SQL injection indicators."
    )
    args_schema: Type[BaseModel] = SecScanArgs  # Pydantic v2 requires annotation

    SECRET_PATTERNS: ClassVar[List[str]] = [
        r"(?i)api[-_ ]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
        r"(?i)secret\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        r"(?i)password\s*[:=]\s*['\"][^'\"]{4,}['\"]",
        r"(?i)bearer\s+[A-Za-z0-9\._\-]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)-----BEGIN (RSA|EC) PRIVATE KEY-----",
    ]
    SQLI_PATTERNS: ClassVar[List[str]] = [
        r"(?i)SELECT\s+.+\s+FROM\s+.+\s*\+\s*",
        r"(?i)(cursor\.execute|execute)\s*\(.+\s*\+\s*",
    ]
    TEXT_EXTS: ClassVar[Set[str]] = {
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".env",
        ".sh", ".bash", ".cfg", ".conf"
    }

    def _run(self, directory: str, max_files: int = 5000) -> str:
        root = Path(directory).resolve()
        if not root.exists():
            return f"Directory does not exist: {root.as_posix()}"

        IGNORE_DIRS = {'.git', '.hg', '.svn', '__pycache__', 'node_modules', '.venv', 'venv', '.mypy_cache'}
        MAX_BYTES = 2 * 1024 * 1024
        NOEXT_OK = {'.env'}  # files with no ext we still want to scan

        secret_res = [re.compile(p) for p in self.SECRET_PATTERNS]
        sqli_res   = [re.compile(p) for p in self.SQLI_PATTERNS]

        findings: List[str] = []
        files_scanned = 0

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

            for fname in filenames:
                if files_scanned >= max_files:
                    break

                path = Path(dirpath) / fname
                ext = path.suffix.lower()

                # Filter by extension (or allow certain no-ext files like .env)
                if ext:
                    if ext not in self.TEXT_EXTS:
                        continue
                else:
                    if fname not in NOEXT_OK:
                        continue

                try:
                    if path.stat().st_size > MAX_BYTES:
                        continue
                except Exception:
                    continue

                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                for i, line in enumerate(text.splitlines(), start=1):
                    for rx in secret_res:
                        for m in rx.finditer(line):
                            findings.append(f"[secret?] {path.as_posix()}:{i} -> {m.group(0)[:200]}")
                    for rx in sqli_res:
                        for m in rx.finditer(line):
                            findings.append(f"[sqli?]   {path.as_posix()}:{i} -> {m.group(0)[:200]}")

                files_scanned += 1

            if files_scanned >= max_files:
                break

        prefix = f"Scanned {files_scanned} files under {root.as_posix()}."
        if not findings:
            return f"No obvious hardcoded secrets or naive SQL-concat patterns found.\n{prefix}"

        # Deduplicate and cap output
        seen = set()
        uniq: List[str] = []
        for f in findings:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
            if len(uniq) >= 500:
                break

        return (f"Potential findings (showing {len(uniq)} of {len(findings)}); {prefix}\n"
                + "\n".join(uniq))


# =========================
# Repo Search Tool (Retriever)
# =========================

def _coerce_query_to_str(value: Any) -> str:
    """Be liberal in what you accept; ensure a plain string reaches the retriever."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        q = value.get("query")
        if isinstance(q, str):
            return q
    try:
        import json
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, dict):
            q = parsed.get("query")
            if isinstance(q, str):
                return q
    except Exception:
        pass
    raise ValueError("`query` must be a plain string (e.g., 'README.md OR architecture').")


def create_repo_analysis_agents(llm, retriever):
    # Args schema for the repo search tool
    class RepoSearchArgs(BaseModel):
        query: str = Field(..., description="Search query over the repo chunks (plain text).")
        model_config = ConfigDict(extra="ignore")

    # Actual tool implementation
    class RepoSearchTool(BaseTool):
        name: str = "Repository Code Search"
        description: str = ("Searches the repository's codebase for relevant chunks. "
                            "Pass a plain string for `query`, e.g., 'README.md OR architecture OR design'.")
        args_schema: Type[BaseModel] = RepoSearchArgs  # Pydantic v2 requires annotation

        _retriever: Any = PrivateAttr()

        def __init__(self, retriever: Any):
            super().__init__()
            self._retriever = retriever

        def _run(self, query: str) -> str:
            try:
                q = _coerce_query_to_str(query)
                docs = self._retriever.get_relevant_documents(q)
                if not docs:
                    return "No relevant chunks."
                return "\n\n".join(getattr(d, "page_content", str(d))[:1200] for d in docs[:6])
            except Exception as e:
                return f"Retriever error: {e}"

    # Instantiate tools
    retrieval_tool = RepoSearchTool(retriever=retriever)
    dependency_tool = DependencyAnalysisTool()
    security_tool = ManualCodeReviewTool()
    enum_tool = EnumerateFilesTool()
    read_tool = ReadFileChunkedTool()

    # Agents
    code_summarizer = Agent(
        role="Principal Code Summarizer",
        goal=("Produce a thorough, per-file analysis of the repository, then summarize the system."),
        backstory=(
            "You are an expert software architect. "
            "You MUST first enumerate all files, then read each file page-by-page until the end. "
            "For every file, write a concise mini-review including PURPOSE, KEY ELEMENTS (functions/classes), I/O, and NOTES. "
            "If a file has multiple pages, continue calling 'Read File (Chunked)' with page=NEXT until PAGE==TOTAL."
        ),
        verbose=True,
        llm=llm,
        tools=[enum_tool, read_tool, retrieval_tool],
        max_iter=60,  # enough room to iterate through many files/pages
        allow_delegation=False,
        respect_context_window=True,
    )

    dependency_analyst = Agent(
        role="Software Dependency Analyst",
        goal="Identify and list third-party dependencies (grouped by ecosystem) from dependency files.",
        backstory=(
            "You analyze dependency files across the repo. "
            "Highlight risky or outdated ecosystems when visible."
        ),
        verbose=True,
        llm=llm,
        tools=[dependency_tool],
        max_iter=6,
        allow_delegation=False,
    )

    # NOTE: give the security agent both tools so it can prove access and then scan
    security_auditor = Agent(
        role="Code Security Auditor",
        goal=(
            "Perform a real scan of the repository and list potential hardcoded secrets or naive SQL concatenations. "
            "Include concrete file:line evidence or clearly state '0 findings' with the number of files scanned."
        ),
        backstory=(
            "You are a hands-on security auditor. "
            "First enumerate files to prove access and count candidates, then run the code review scan. "
            "Do not output generic disclaimers—produce concrete results."
        ),
        verbose=True,
        llm=llm,
        tools=[enum_tool, security_tool],  # enumerate then scan
        max_iter=8,
        allow_delegation=False,
        respect_context_window=True,
    )

    documentation_generator = Agent(
        role="Technical Writer",
        goal=(
            "Synthesize the per-file reviews, dependency list, and security findings into a single report. "
            "Include: Summary, Architecture & Data Flow, Per-file Highlights, Dependencies, Security, Recommendations."
        ),
        backstory=("You produce clear, actionable engineering reports."),
        verbose=True,
        llm=llm,
        max_iter=4,
        allow_delegation=False,
    )

    return {
        "summarizer": code_summarizer,
        "dependency_analyst": dependency_analyst,
        "security_auditor": security_auditor,
        "documentation_generator": documentation_generator,
    }