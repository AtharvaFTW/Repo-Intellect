import os
import re
from typing import Optional, List, Type,ClassVar,Set,Any
from crewai import Agent
from crewai.tools import BaseTool
from llm_setup import get_llm
from pydantic import BaseModel,Field, ConfigDict, PrivateAttr
from pathlib import Path

class DepArgs(BaseModel):
    directory: str = Field(..., description="Path to repository root to scan.")
    file: Optional[str] = Field(
        default=None,
        description="Optional file name (e.g., requirements.txt, package.json)")
    model_config = ConfigDict(extra="ignore")
class DependencyAnalysisTool(BaseTool):
    name: str = "Dependency Anaylsis Tool"
    description: str = "Parses common dependency files like requirements.txt or package.json to a list of project dependencies."
    args_schema: type[BaseModel] = DepArgs

    def _run(self, directory: str, max_files: int = 2000) -> str:
        root = Path(directory).resolve()
        if not root.exists():
            return f"Directory does not exist: {root.as_posix()}"

        # Directories to ignore during walk
        IGNORE_DIRS = {'.git', '.hg', '.svn', '__pycache__', 'node_modules', '.venv', 'venv', '.mypy_cache'}
        # Skip very large files (2 MB)
        MAX_BYTES = 2 * 1024 * 1024
        # Some config files have no extension but should be scanned
        NOEXT_OK = {'.env'}

        # Compile patterns once
        secret_res = [re.compile(p) for p in self.SECRET_PATTERNS]
        sqli_res   = [re.compile(p) for p in self.SQLI_PATTERNS]

        findings: List[str] = []
        files_scanned = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # prune ignored dirs in-place (speeds up traversal)
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

                # Skip huge files / unreadable paths
                try:
                    if path.stat().st_size > MAX_BYTES:
                        continue
                except Exception:
                    continue

                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                # Scan line-by-line for accurate line numbers
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

        if not findings:
            return (f"No obvious hardcoded secrets or naive SQL-concat patterns found.\n"
                    f"Scanned {files_scanned} files under {root.as_posix()}.")

        # Deduplicate and cap output
        seen = set()
        uniq: List[str] = []
        for f in findings:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
            if len(uniq) >= 500:
                break

        return (f"Potential findings (showing {len(uniq)} of {len(findings)}; "
                f"scanned {files_scanned} files under {root.as_posix()}):\n" + "\n".join(uniq))
# ---------- NEW: simple security scan tool ----------
class SecScanArgs(BaseModel):
     directory: str = Field(..., description="Repo root directory to scan")
     max_files: int = Field(2000, description="Safety cap on number of files to scan")
     model_config = ConfigDict(extra="ignore")

class ManualCodeReviewTool(BaseTool):
    name: str = "Manual Code Review (for hardcoded secrets)"
    description: str = ("Greps source files for common secret patterns (keys, tokens, passwords) "
                         "and simple SQL injection indicators.")
    args_schema: type[BaseModel] = SecScanArgs

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
    TEXT_EXTS: ClassVar[Set[str]] = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".env"}

    def _run(self, directory: str, file: Optional[str] = None) -> str:
        import os
        from pathlib import Path
        from typing import Optional, List

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

        def read_text(p: Path) -> Optional[str]:
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None

        # Case 1: specific file explicitly requested (search for it anywhere under root)
        if file:
            wanted = file.lower()
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
                for fname in filenames:
                    if fname.lower() == wanted:
                        p = Path(dirpath) / fname
                        eco = dependency_files.get(fname, "Unknown ecosystem")
                        content = read_text(p)
                        if content:
                            return f"--- {eco} Dependencies ({p.relative_to(root)}) ---\n{content}\n"
                        else:
                            return f"Found {fname} at {p.as_posix()} but could not read it."
            return f"{file} not found under {root.as_posix()}."

        # Case 2: scan for all known dependency files anywhere under root
        found: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for fname in filenames:
                lname = fname.lower()
                if lname in dependency_files:
                    p = Path(dirpath) / fname
                    eco = dependency_files.get(fname, "Unknown ecosystem")
                    content = read_text(p)
                    if content:
                        rel = p.relative_to(root)
                        found.append(f"--- {eco} Dependencies ({rel}) ---\n{content}\n")

        if not found:
            return f"No common dependency files found under {root.as_posix()}."
        return "\n".join(found)



def create_repo_analysis_agents(llm, retriever):
    # small wrapper tool so the summarizer can query your retriever
    class RepoSearchArgs(BaseModel):
        query: str = Field(..., description="Search query over the repo chunks")
        model_config = ConfigDict(extra="ignore")

    class RepoSearchTool(BaseTool):
        name: str = "Repository Code Search"
        description: str = "Searches the repository's codebase for relevant chunks."
        args_schema: type[BaseModel] = RepoSearchArgs

    # declare a private attribute for pydantic v2
        _retriever: Any = PrivateAttr()

        def __init__(self, retriever: Any):
            super().__init__()
            self._retriever = retriever

        def _run(self, query: str) -> str:
            try:
                docs = self._retriever.get_relevant_documents(query)
                if not docs:
                    return "No relevant chunks."
                return "\n\n".join(getattr(d, "page_content", str(d))[:1200] for d in docs[:6])
            except Exception as e:
                return f"Retriever error: {e}"

    retrieval_tool = RepoSearchTool(retriever=retriever)
    dependency_tool = DependencyAnalysisTool()
    security_tool = ManualCodeReviewTool()

    code_summarizer=Agent(
        role="Principal Code Summarizer",
        goal="Generate a high-level summary of the repository's purpose, architecture, and key components.",
        backstory=("""
            You are an expert software architect with years of experience in analyzing large codebases.
            Your task is to provide a clear and concise overview that helps developers quickly underand the project."""),
        verbose=True,
        llm=llm,
        tools=[retrieval_tool],
        max_iter=3,
        allow_delegation=False)
    
    dependency_analyst=Agent(
        role="Software Dependency Analyst",
        goal="Identify and list all external dependencies of the project",
        backstory=("""
            You are a meticulous software engineer specializing in supply chain security.
            Your job is to find and analyze all the dependency files to create a comprehensive list of third-party libraries."""),
            verbose=True,
            llm=llm,
            tools=[dependency_tool],
            max_iter=3,
            allow_delegation=False)
    
    security_auditor=Agent(
        role="Code Security Auditor",
        goal="Perform a basic security scan of the codebase to identify common vulnerabilities.",
        backstory=("""
            You are a sequrity expert with a keen eye for vulnerabilities.
            You scan code for potential issues like harcorded secrets and SQL injection patterns, providing a preliminary security assessment."""),
            verbose=True,
            llm=llm,
            tools=[security_tool],
            max_iter=3,
            allow_delegation=False)
    
    documentation_generator=Agent(
        role="Technical Writer",
        goal="Synthesize the analysis from other agents into a comprehensive and well-structured final report",
        backstory=("""
            You are a professional techinal writer known for you ability to distill complex technical information
            into a clear, easy-to-understand documentation. You will take findings from your team and assemble them into a final report."""),
            verbose=True,
            llm=llm,
            max_iter=3,
            allow_delegation=False)
    
    return {
        "summarizer":code_summarizer,
        "dependency_analyst":dependency_analyst,
        "security_auditor":security_auditor,
        "documentation_generator":documentation_generator
    }
