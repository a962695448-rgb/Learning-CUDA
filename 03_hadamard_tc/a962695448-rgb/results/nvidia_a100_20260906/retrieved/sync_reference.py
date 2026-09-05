#!/usr/bin/env python3
"""Transport official GitHub archives and restore their original shallow Git commit.

Usage: python server_git_sync.py manifest.json [--probe-only] [--json LOG.json]
Manifest: {"repositories": [{"owner": "OWNER", "name": "REPO",
  "branch": "BRANCH", "commit": "40_HEX", "tree_sha": "40_HEX",
  "commit_object_b64": "BASE64_OF_GIT_CAT_FILE_COMMIT_BYTES",
  "upstream": "https://github.com/UPSTREAM/REPO.git"}]}

Targets for this A100 task are inside the script directory only. The original commit
bytes are never edited. No git commit command or remote push is performed.
"""
import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import uuid

BASE = Path(__file__).resolve().parent
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 200000


def utc():
    return datetime.now(timezone.utc).isoformat()


def bounded(path, base):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(base).resolve()):
        raise RuntimeError(f"Refusing path outside {base}: {resolved}")
    return resolved


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


class Log:
    def __init__(self, output, base):
        self.base = Path(base).resolve()
        self.output = bounded(output, self.base)
        self.report = {"status": "RUNNING", "started_utc": utc(), "commands": [], "repositories": []}

    def save(self):
        bounded(self.output.parent, self.base).mkdir(parents=True, exist_ok=True)
        temporary = bounded(self.output.with_suffix(self.output.suffix + ".tmp"), self.base)
        temporary.write_text(json.dumps(self.report, indent=2) + "\n")
        temporary.replace(self.output)

    def git(self, directory, *arguments, data=None, required=True):
        command = ["git", "-C", str(directory), *arguments]
        # Disable inherited filters/hooks/config; Git operations here are local.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                            "GIT_TERMINAL_PROMPT": "0", "GIT_LITERAL_PATHSPECS": "1",
                            "GIT_OPTIONAL_LOCKS": "0"})
        entry = {"argv": command, "started_utc": utc()}
        self.report["commands"].append(entry)
        self.save()
        try:
            result = subprocess.run(command, input=data, capture_output=True, env=environment, timeout=60)
            entry.update({"returncode": result.returncode,
                          "stdout": result.stdout.decode("utf-8", "replace"),
                          "stderr": result.stderr.decode("utf-8", "replace")})
            self.save()
            if required and result.returncode:
                raise RuntimeError(f"git {arguments[0]} failed with exit {result.returncode}; see command log")
            return result.returncode, entry["stdout"].strip()
        except Exception as error:
            entry["exception"] = f"{type(error).__name__}: {error}"
            self.save()
            raise


def validate_item(item):
    result = dict(item)
    result["name"] = item.get("name", item.get("repo"))
    result["tree_sha"] = item.get("tree_sha", item.get("expected_tree_sha"))
    for key in ("owner", "name"):
        value = result.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", "..", ".git"):
            raise ValueError(f"Invalid repository {key}")
    for key in ("commit", "tree_sha"):
        if not isinstance(result.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", result[key]):
            raise ValueError(f"{key} must be a lowercase SHA-1 with 40 hexadecimal characters")
    if not isinstance(result.get("branch"), str) or not result["branch"]:
        raise ValueError("A nonempty branch is required")
    raw = base64.b64decode(result["commit_object_b64"], validate=True)
    identity = hashlib.sha1(b"commit " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if identity != result["commit"]:
        raise ValueError(f"Original commit object hashes to {identity}, not {result['commit']}")
    if raw.split(b"\n", 1)[0] != b"tree " + result["tree_sha"].encode("ascii"):
        raise ValueError("Original commit object does not name the expected tree")
    if result.get("upstream"):
        parsed = urllib.parse.urlsplit(result["upstream"])
        if (parsed.scheme != "https" or parsed.netloc != "github.com" or parsed.query or parsed.fragment
                or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", parsed.path)):
            raise ValueError("upstream must be a public HTTPS github.com owner/repository URL")
    result["raw_commit"] = raw
    result["archive_url"] = f"https://codeload.github.com/{result['owner']}/{result['name']}/tar.gz/{result['commit']}"
    return result


def open_archive(url):
    request = urllib.request.Request(url, headers={"User-Agent": "InfiniTensor-ArchiveSync/1.0"})
    response = urllib.request.urlopen(request, context=ssl.create_default_context(), timeout=25)
    final = urllib.parse.urlsplit(response.geturl())
    if response.status != 200 or final.scheme != "https" or final.netloc != "codeload.github.com":
        response.close()
        raise RuntimeError("Official codeload request did not return HTTP 200 on verified HTTPS codeload.github.com")
    return response


def download_archive(item, base, record):
    cache = bounded(base / ".git-sync-cache", base)
    cache.mkdir(parents=True, exist_ok=True)
    archive = bounded(cache / f"{item['owner']}-{item['name']}-{item['commit']}.tar.gz", base)
    if archive.exists():
        if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise RuntimeError("Invalid cached archive")
        record["archive_reused"] = True
    else:
        temporary = bounded(archive.with_suffix(archive.suffix + ".part"), base)
        start = time.monotonic()
        count = 0
        with open_archive(item["archive_url"]) as response, temporary.open("wb") as output:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_ARCHIVE_BYTES:
                raise RuntimeError("Archive Content-Length exceeds the 512 MiB limit")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > MAX_ARCHIVE_BYTES or time.monotonic() - start > 180:
                    raise RuntimeError("Archive download exceeded its size/time limit")
                output.write(block)
        if count < 2:
            raise RuntimeError("Empty or incomplete archive response")
        temporary.replace(archive)
        record["archive_reused"] = False
    record.update({"archive": str(archive), "archive_bytes": archive.stat().st_size,
                   "archive_sha256": sha256(archive),
                   "archive_hash_note": "Observed transfer digest; authenticity is checked by reconstructed Git tree/commit hashes"})
    return archive


def extract_sources(archive_path, staging):
    source_files = []
    root_name = None
    seen = set()
    extracted_size = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for index, original in enumerate(archive):
            if index >= MAX_MEMBERS:
                raise RuntimeError("Archive member count exceeds limit")
            if "\\" in original.name or "\0" in original.name:
                raise RuntimeError("Unsupported/unsafe archive pathname")
            parts = PurePosixPath(original.name).parts
            if not parts or PurePosixPath(original.name).is_absolute() or ".." in parts:
                raise RuntimeError("Unsafe archive pathname")
            if root_name is None:
                root_name = parts[0]
            if parts[0] != root_name:
                raise RuntimeError("Archive must contain exactly one top-level directory")
            if len(parts) == 1:
                if not original.isdir():
                    raise RuntimeError("Archive top-level member is not a directory")
                continue
            relative = PurePosixPath(*parts[1:])
            if any(part.lower() == ".git" for part in relative.parts):
                raise RuntimeError("Archive must not supply Git administrative files")
            if not (original.isfile() or original.isdir() or original.issym()):
                raise RuntimeError("Archive contains a hard link, device, FIFO, or unsupported member")
            if str(relative) in seen:
                raise RuntimeError("Duplicate archive pathname")
            seen.add(str(relative))
            if original.size < 0:
                raise RuntimeError("Invalid archive member size")
            extracted_size += original.size
            if extracted_size > MAX_EXTRACTED_BYTES:
                raise RuntimeError("Archive expands beyond the 2 GiB limit")
            member = copy.copy(original)
            member.name = str(relative)
            # Python's data filter rejects absolute/escaping symlinks and paths,
            # strips special mode bits, and never restores archive ownership.
            archive.extract(member, path=staging, filter="data")
            if original.isfile() or original.issym():
                source_files.append(str(relative))
    if not source_files:
        raise RuntimeError("Archive contained no source files")
    return source_files


def restore_git(staging, item, source_files, logger):
    logger.git(staging, "init", "--quiet")
    logger.git(staging, "config", "core.autocrlf", "false")
    logger.git(staging, "config", "core.filemode", "true")
    # Explicit NUL-delimited literal paths prevent glob interpretation, and -f
    # preserves archived sources that are ignored by the repository's .gitignore.
    path_list = staging / ".git" / "archive-source-paths"
    path_list.write_bytes(b"".join(os.fsencode(path) + b"\0" for path in source_files))
    logger.git(staging, "add", "--force", "--pathspec-from-file=" + str(path_list), "--pathspec-file-nul")
    _, actual_tree = logger.git(staging, "write-tree")
    if actual_tree != item["tree_sha"]:
        raise RuntimeError(f"Archive tree mismatch: reconstructed {actual_tree}, expected {item['tree_sha']}. "
                           "No commit will be fabricated; export-ignore/export-subst or gitlinks may explain a mismatch.")
    _, actual_commit = logger.git(staging, "hash-object", "-t", "commit", "-w", "--stdin", data=item["raw_commit"])
    if actual_commit != item["commit"]:
        raise RuntimeError("Git hash-object did not reproduce the original commit")
    (staging / ".git" / "shallow").write_text(actual_commit + "\n")
    logger.git(staging, "update-ref", "refs/heads/" + item["branch"], actual_commit)
    logger.git(staging, "symbolic-ref", "HEAD", "refs/heads/" + item["branch"])
    origin = f"https://github.com/{item['owner']}/{item['name']}.git"
    logger.git(staging, "remote", "add", "origin", origin)
    if item.get("upstream"):
        logger.git(staging, "remote", "add", "upstream", item["upstream"])
    _, status = logger.git(staging, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("Reconstructed checkout is not clean")
    logger.git(staging, "fsck", "--connectivity-only", "--no-reflogs")
    return actual_tree, actual_commit


def existing_clean(target, item, logger):
    if not target.exists():
        return False
    if not target.is_dir():
        raise RuntimeError("Existing target is not a directory")
    if not any(target.iterdir()):
        return False
    if not (target / ".git").is_dir():
        raise RuntimeError("Refusing nonempty target without its own Git directory")
    _, top = logger.git(target, "rev-parse", "--show-toplevel")
    _, head = logger.git(target, "rev-parse", "HEAD")
    _, tree = logger.git(target, "rev-parse", "HEAD^{tree}")
    _, status = logger.git(target, "status", "--porcelain", "--untracked-files=all")
    if Path(top).resolve() != target or head != item["commit"] or tree != item["tree_sha"] or status:
        raise RuntimeError("Refusing nonempty existing target: it is not the same clean commit/tree")
    return True


def synchronize(item, logger, probe_only=False):
    item = validate_item(item)
    base = logger.base
    target = bounded(base / item["name"], base)
    if target == base:
        raise RuntimeError("Target must be a repository child directory")
    record = {"owner": item["owner"], "name": item["name"], "branch": item["branch"],
              "commit": item["commit"], "tree_sha": item["tree_sha"], "target": str(target),
              "archive_url": item["archive_url"], "status": "RUNNING"}
    logger.report["repositories"].append(record)
    logger.save()
    try:
        base.mkdir(parents=True, exist_ok=True)
        logger.git(base, "check-ref-format", "--branch", item["branch"])
        if probe_only:
            start = time.monotonic()
            with open_archive(item["archive_url"]) as response:
                prefix = response.read(65536)
            if not prefix.startswith(b"\x1f\x8b"):
                raise RuntimeError("Codeload response did not start with a gzip archive header")
            record.update({"status": "PROBE_PASS", "prefix_bytes_read": len(prefix),
                           "elapsed_seconds": time.monotonic() - start,
                           "note": "Connectivity probe only; full archive/tree has not been verified"})
        elif existing_clean(target, item, logger):
            record.update({"status": "REUSED", "note": "Same clean commit/tree; existing directory not modified"})
        else:
            archive = download_archive(item, base, record)
            staging_parent = bounded(base / ".git-sync-staging", base)
            staging_parent.mkdir(parents=True, exist_ok=True)
            staging = bounded(staging_parent / (item["name"] + "-" + uuid.uuid4().hex), base)
            staging.mkdir(exist_ok=False)
            record["staging"] = str(staging)
            logger.save()
            sources = extract_sources(archive, staging)
            tree, commit = restore_git(staging, item, sources, logger)
            # Recheck the final destination before publishing; never replace data
            # created by another process while transfer/verification was running.
            if target.exists():
                if not target.is_dir() or any(target.iterdir()):
                    raise RuntimeError("Target became nonempty during transfer; refusing to replace it")
                target.rmdir()  # An explicitly checked empty directory only.
            staging.rename(target)
            record.update({"status": "RESTORED", "verified_commit": commit, "verified_tree": tree,
                           "source_files": len(sources), "shallow_boundary": commit})
        logger.save()
        print(json.dumps(record), flush=True)
    except Exception as error:
        record.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}"})
        logger.save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--json", default=str(BASE / "logs/server_git_sync.json"))
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info < (3, 12):
        print(json.dumps({"status": "ERROR", "error": "Run inside the Linux server with Python >=3.12"}))
        return 2
    logger = None
    try:
        logger = Log(args.json, BASE)
        data = json.loads(Path(args.manifest).read_text())
        items = data.get("repositories") if isinstance(data, dict) else data
        if not isinstance(items, list) or not items:
            raise ValueError("manifest must contain a nonempty repositories list")
        names = [validate_item(item)["name"] for item in items]
        if len(set(names)) != len(names):
            raise ValueError("manifest contains duplicate target repository names")
        for item in items:
            synchronize(item, logger, args.probe_only)
        logger.report.update({"status": "PROBE_PASS" if args.probe_only else "PASS", "finished_utc": utc()})
        code = 0
    except Exception as error:
        if logger is None:
            print(json.dumps({"status": "ERROR", "error": f"{type(error).__name__}: {error}"}))
            return 2
        logger.report.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}", "finished_utc": utc()})
        code = 2
    try:
        logger.save()
    except Exception as error:
        print(json.dumps({"status": "ERROR", "error": f"Cannot write log: {error}"}))
        return 2
    print(json.dumps({"status": logger.report["status"], "json": str(logger.output),
                      "error": logger.report.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
