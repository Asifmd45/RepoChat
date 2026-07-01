import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from github import Github
from dotenv import load_dotenv
import time
import base64

load_dotenv()

SUPPORTED_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".md", ".txt",
    ".yaml", ".yml", ".json", ".toml", ".env.example", ".sh"
]

SKIP_DIRS = ["node_modules", ".git", "dist", "build", "__pycache__", ".next"]

MAX_FILE_SIZE = 100 * 1024


def parse_repo_url(url: str):
    url = url.strip().rstrip("/")
    parts = url.replace("https://", "").replace("http://", "").split("/")
    owner, repo = parts[1], parts[2]
    return f"{owner}/{repo}"

MAX_SAFE_FILES = 600  # threshold based on your real AlgoBuddy/FastAPI experience

def check_repo_size(repo) -> dict:
    """Quick check of repo size before committing to full ingestion."""
    branch = repo.default_branch
    tree = repo.get_git_tree(branch, recursive=True)

    valid_count = 0
    for item in tree.tree:
        if item.type != "blob":
            continue
        if any(skip in item.path for skip in SKIP_DIRS):
            continue
        ext = os.path.splitext(item.path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        valid_count += 1

    return {
        "file_count": valid_count,
        "is_safe": valid_count <= MAX_SAFE_FILES
    }


def fetch_repo_files(repo) -> list[dict]:
    documents = []

    t1 = time.time()
    branch = repo.default_branch
    tree = repo.get_git_tree(branch, recursive=True)
    print(f"  Tree fetch: {time.time() - t1:.2f}s")

    valid_files = []
    for item in tree.tree:
        if item.type != "blob":
            continue
        if any(skip in item.path for skip in SKIP_DIRS):
            continue
        ext = os.path.splitext(item.path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        valid_files.append(item)

    print(f"  Valid files to fetch: {len(valid_files)}")
    t2 = time.time()

    def fetch_content(item):
        try:
            blob = repo.get_git_blob(item.sha)
            text = base64.b64decode(blob.content).decode("utf-8", errors="ignore")
            if len(text.encode('utf-8')) > MAX_FILE_SIZE:
                return None
            ext = os.path.splitext(item.path)[1].lower()
            return {
                "content": text,
                "metadata": {
                    "type": "code" if ext != ".md" else "markdown",
                    "source": item.path,
                    "file_name": os.path.basename(item.path),
                    "extension": ext
                }
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_content, item) for item in valid_files]
        for future in as_completed(futures):
            result = future.result()
            if result:
                documents.append(result)

    print(f"  Content fetch: {time.time() - t2:.2f}s")
    return documents


def fetch_issues(repo) -> list[dict]:
    documents = []
    issues = list(repo.get_issues(state="open"))

    def process_issue(issue):
        try:
            if issue.pull_request:
                return None
            body = issue.body or ""
            assignees = [a.login for a in issue.assignees]
            labels = [l.name for l in issue.labels]
            text = f"Issue #{issue.number}: {issue.title}\n\n{body}"
            return {
                "content": text,
                "metadata": {
                    "type": "issue",
                    "source": issue.html_url,
                    "issue_number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "labels": ", ".join(labels),
                    "assignees": ", ".join(assignees) if assignees else "unassigned"
                }
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_issue, issue) for issue in issues]
        for future in as_completed(futures):
            result = future.result()
            if result:
                documents.append(result)

    return documents


def fetch_pull_requests(repo) -> list[dict]:
    documents = []
    prs = list(repo.get_pulls(state="open"))

    def process_pr(pr):
        try:
            body = pr.body or ""
            text = f"PR #{pr.number}: {pr.title}\n\n{body}"
            return {
                "content": text,
                "metadata": {
                    "type": "pull_request",
                    "source": pr.html_url,
                    "pr_number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "author": pr.user.login
                }
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_pr, pr) for pr in prs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                documents.append(result)

    return documents


class RepoTooLargeError(Exception):
    pass


def ingest_repo(repo_url: str) -> list[dict]:
    token = os.getenv("GITHUB_TOKEN")
    g = Github(token)
    repo_path = parse_repo_url(repo_url)
    repo = g.get_repo(repo_path)

    print(f"Fetching repo: {repo.full_name}")

    size_check = check_repo_size(repo)
    print(f"  File count check: {size_check['file_count']} files")

    if not size_check["is_safe"]:
        raise RepoTooLargeError(
            f"This repository has {size_check['file_count']} files, which exceeds the "
            f"{MAX_SAFE_FILES} file limit for reliable loading. Large repos may hit "
            f"GitHub API rate limits or take several minutes to process. "
            f"Try a smaller repo or a specific subfolder instead."
        )

    files = fetch_repo_files(repo)
    print(f"  Files fetched: {len(files)}")

    issues = fetch_issues(repo)
    print(f"  Issues fetched: {len(issues)}")

    prs = fetch_pull_requests(repo)
    print(f"  PRs fetched: {len(prs)}")

    all_docs = files + issues + prs
    print(f"  Total documents: {len(all_docs)}")

    return all_docs


if __name__ == "__main__":
    start = time.time()
    docs = ingest_repo("https://github.com/pankajsingh34/algobuddy")
    end = time.time()

    for doc in docs[:3]:
        print("\n---")
        print(f"Type: {doc['metadata']['type']}")
        print(f"Source: {doc['metadata']['source']}")
        print(f"Preview: {doc['content'][:200]}")

    print(f"\n⏱️ Total time: {end - start:.2f} seconds")