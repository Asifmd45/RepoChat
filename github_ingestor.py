import os
from github import Github
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".md", ".txt",
    ".yaml", ".yml", ".json", ".toml", ".env.example", ".sh"
]

MAX_FILE_SIZE = 100 * 1024  # 100KB — skip huge generated files


def parse_repo_url(url: str):
    # handles https://github.com/owner/repo or github.com/owner/repo
    url = url.strip().rstrip("/")
    parts = url.replace("https://", "").replace("http://", "").split("/")
    owner, repo = parts[1], parts[2]
    return f"{owner}/{repo}"


def fetch_repo_files(repo) -> list[dict]:
    documents = []
    contents = repo.get_contents("")

    # skip node_modules and other noise folders
    SKIP_DIRS = ["node_modules", ".git", "dist", "build", "__pycache__", ".next"]

    while contents:
        file = contents.pop(0)
        # add this check
        if any(skip in file.path for skip in SKIP_DIRS):
            continue
        if file.type == "dir":
            contents.extend(repo.get_contents(file.path))
        else:
            ext = os.path.splitext(file.name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            if file.size > MAX_FILE_SIZE:
                continue
            try:
                text = file.decoded_content.decode("utf-8", errors="ignore")
                documents.append({
                    "content": text,
                    "metadata": {
                        "type": "code" if ext != ".md" else "markdown",
                        "source": file.path,
                        "file_name": file.name,
                        "extension": ext
                    }
                })
            except Exception:
                continue

    return documents


def fetch_issues(repo) -> list[dict]:
    documents = []
    issues = repo.get_issues(state="open")

    for issue in issues:
        if issue.pull_request:
            continue  # skip PRs here, handled separately
        body = issue.body or ""
        assignees = [a.login for a in issue.assignees]
        labels = [l.name for l in issue.labels]
        text = f"Issue #{issue.number}: {issue.title}\n\n{body}"

        documents.append({
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
        })

    return documents


def fetch_pull_requests(repo) -> list[dict]:
    documents = []
    prs = repo.get_pulls(state="open")

    for pr in prs:
        body = pr.body or ""
        text = f"PR #{pr.number}: {pr.title}\n\n{body}"

        documents.append({
            "content": text,
            "metadata": {
                "type": "pull_request",
                "source": pr.html_url,
                "pr_number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "author": pr.user.login
            }
        })

    return documents


def ingest_repo(repo_url: str) -> list[dict]:
    token = os.getenv("GITHUB_TOKEN")
    g = Github(token)
    repo_path = parse_repo_url(repo_url)
    repo = g.get_repo(repo_path)

    print(f"Fetching repo: {repo.full_name}")

    files = fetch_repo_files(repo)
    print(f"  Files fetched: {len(files)}")

    issues = fetch_issues(repo)
    print(f"  Issues fetched: {len(issues)}")

    prs = fetch_pull_requests(repo)
    print(f"  PRs fetched: {len(prs)}")

    all_docs = files + issues + prs
    print(f"  Total documents: {len(all_docs)}")

    return all_docs


# Quick test
if __name__ == "__main__":
    docs = ingest_repo("https://github.com/Asifmd45/Advanced-RAG-Pipeline")
    for doc in docs[:3]:
        print("\n---")
        print(f"Type: {doc['metadata']['type']}")
        print(f"Source: {doc['metadata']['source']}")
        print(f"Preview: {doc['content'][:200]}")