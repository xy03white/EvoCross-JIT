import pandas as pd
import subprocess
import os
from tqdm import tqdm
import csv

DATASET_FILE = "./dataset/jdt_train.csv"
OUTPUT_FILE = "./dataset/enriched/jdt_train.csv"

REPO_BASE = "./repos/jdt"
REPO_PATHS = [
    os.path.join(REPO_BASE, name)
    for name in os.listdir(REPO_BASE)
    if os.path.isdir(os.path.join(REPO_BASE, name))
]
for p in REPO_PATHS:
    print("  ", p)

def get_lang(file_path):
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    lang_map = {
        'py': 'python', 'java': 'java', 'go': 'go',
        'cpp': 'cpp', 'cc': 'cpp', 'cxx': 'cpp',
        'js': 'javascript', 'ts': 'javascript',
        'rb': 'ruby', 'php': 'php', 'cs': 'c_sharp', 'c': 'c'
    }
    return lang_map.get(ext, 'unknown')

def run_git(cmd, repo_path):
    try:
        return subprocess.check_output(
            cmd, cwd=repo_path, stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="ignore"
        )
    except subprocess.CalledProcessError:
        return None

def find_commit_repo(commit_id):
    for repo_path in REPO_PATHS:
        result = subprocess.run(
            ["git", "cat-file", "-e", commit_id],
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            return repo_path
        try:
            result = subprocess.run(
                ["git", "branch", "-a", "--contains", commit_id],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                return repo_path
        except:
            continue
    return None

def get_changed_files(commit_id, repo_path):
    result = run_git(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_id], repo_path)
    if not result:
        return None
    
    code_files = []
    for file in result.strip().split('\n'):
        lang = get_lang(file)
        if lang != 'unknown':
            code_files.append(file)
    
    return code_files if code_files else None

def get_old_file(repo_path, commit_id, file_path):
    return run_git(["git", "show", f"{commit_id}^:{file_path}"], repo_path)

def get_old_diff(repo_path, commit_id, file_path):
    return run_git(["git", "diff", f"{commit_id}^", commit_id, "--", file_path], repo_path)

def combine_old_files(old_files_data):
    if not old_files_data:
        return ""
    
    combined_content = []
    for file_path, content in old_files_data.items():
        combined_content.append(f"# File: {file_path}\n{content}")
    
    return "\n\n".join(combined_content)

def combine_old_diffs(old_diffs_data):
    if not old_diffs_data:
        return ""
    
    combined_diff = []
    for file_path, diff_content in old_diffs_data.items():
        combined_diff.append(f"# File: {file_path}\n{diff_content}")
    
    return "\n\n".join(combined_diff)

def main():
    if not os.path.exists(DATASET_FILE):
        return

    df = pd.read_csv(DATASET_FILE)
    total_commits = len(df)

    processed_ids = set()
    header_written = False
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        processed_df = pd.read_csv(OUTPUT_FILE, usecols=["_id"])
        processed_ids = set(processed_df["_id"].astype(str))
        header_written = True

    found_count = 0
    missing_count = 0
    label_0_count = 0
    label_1_count = 0
    files_count_distribution = {}

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    csv_file = open(OUTPUT_FILE, 'a', newline='', encoding='utf-8')
    writer = None

    for idx, row in tqdm(df.iterrows(), total=total_commits):
        cid = str(row["_id"])

        if cid in processed_ids:
            continue

        repo_path = find_commit_repo(cid)
        if not repo_path:
            missing_count += 1
            continue

        file_paths = get_changed_files(cid, repo_path)
        if not file_paths:
            missing_count += 1
            continue

        old_files_data = {}
        old_diffs_data = {}
        success_files = 0
        
        for file_path in file_paths:
            old_file = get_old_file(repo_path, cid, file_path)
            old_diff = get_old_diff(repo_path, cid, file_path)
            
            if old_file and old_diff:
                old_files_data[file_path] = old_file
                old_diffs_data[file_path] = old_diff
                success_files += 1

        if success_files == 0:
            missing_count += 1
            continue

        found_count += 1
        if row["label"] == 0:
            label_0_count += 1
        else:
            label_1_count += 1
        
        files_count = len(file_paths)
        files_count_distribution[files_count] = files_count_distribution.get(files_count, 0) + 1

        new_row = row.to_dict()
        new_row["repo_path"] = repo_path
        new_row["file_path"] = file_paths[0]
        new_row["lang"] = get_lang(file_paths[0])
        new_row["old_file"] = combine_old_files(old_files_data)
        new_row["old_code_diff"] = combine_old_diffs(old_diffs_data)

        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=list(new_row.keys()))
            if not header_written:
                writer.writeheader()
                header_written = True

        writer.writerow(new_row)
        csv_file.flush()

    csv_file.close()


if __name__ == "__main__":
    main()