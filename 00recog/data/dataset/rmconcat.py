"""
makes the dataset
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


# STOLEN HELPERS SECTION
def remove_duplicates_system_sort(input_filename, output_filename):
    """
    Removes duplicate lines by using the system `sort -u` and writes to output_filename.
    Falls back to a Python implementation if `sort` isn't available.
    """
    try:
        # Run: sort -u input_filename
        # Write stdout directly to output file (no shell redirection needed).
        with open(output_filename, "w", encoding="utf-8", newline="") as out_f:
            subprocess.run(
                ["sort", "-u", input_filename],
                check=True,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
            )
        print(f"Duplicates removed and saved to {output_filename} using system sort.")
        return

    except FileNotFoundError:
        # `sort` not installed / not on PATH -> fallback
        print("System 'sort' command not found. Falling back to Python dedupe.")

    except subprocess.CalledProcessError as e:
        # sort ran but failed (bad file path, permission, etc.)
        err = (e.stderr or "").strip()
        print(f"System sort failed. Falling back to Python dedupe. Error: {err}")

    # --- Python fallback (preserves input order; not sorted) ---
    seen = set()
    with open(input_filename, "r", encoding="utf-8", errors="replace") as in_f, open(
        output_filename, "w", encoding="utf-8", newline=""
    ) as out_f:
        for line in in_f:
            if line not in seen:
                seen.add(line)
                out_f.write(line)

    print(f"Duplicates removed and saved to {output_filename} using Python fallback.")


def remove_blacklisted_from_output(
    output_file: str | Path,
    french_blacklist: str | Path,
    german_blacklist: str | Path,
    *,
    out_file: str | Path | None = None,
    in_place: bool = False,
    case_insensitive: bool = False,
    strip_whitespace: bool = True,
    keep_empty_lines: bool = False,
) -> Path:
    """
    Removes any line from output_file that appears in either french_blacklist or german_blacklist.

    Assumes one entry per line.

    - If in_place=True: overwrites output_file safely (atomic replace).
    - Else: writes to out_file (or output_file.with_suffix(".filtered.txt") if not provided).

    Returns the path written.
    """
    output_path = Path(output_file)
    fr_path = Path(french_blacklist)
    de_path = Path(german_blacklist)

    def norm(s: str) -> str:
        if strip_whitespace:
            s = s.strip()
        if case_insensitive:
            s = s.lower()
        return s

    # Load both blacklists into one set
    blacklist: set[str] = set()
    for p in (fr_path, de_path):
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                k = norm(line)
                if k or keep_empty_lines:
                    blacklist.add(k)

    # Decide where to write
    if in_place:
        target_path = output_path
    else:
        target_path = (
            Path(out_file)
            if out_file is not None
            else output_path.with_suffix(".filtered.txt")
        )

    # Write safely: if in_place, write temp then atomic replace
    if in_place:
        fd, tmp_name = tempfile.mkstemp(
            prefix=output_path.name + ".", dir=str(output_path.parent)
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            with output_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as inf, tmp_path.open("w", encoding="utf-8", newline="") as outf:
                for line in inf:
                    k = norm(line)
                    if not k and not keep_empty_lines:
                        continue
                    if k not in blacklist:
                        # preserve original line formatting
                        outf.write(line)
            tmp_path.replace(output_path)  # atomic on same filesystem
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return output_path

    # Non-in-place write
    with output_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as inf, target_path.open("w", encoding="utf-8", newline="") as outf:
        for line in inf:
            k = norm(line)
            if not k and not keep_empty_lines:
                continue
            if k not in blacklist:
                outf.write(line)

    return target_path


def delete_file(file_path):
    """
    delete pathlib file
    """
    if file_path.is_file():
        try:
            file_path.unlink()
        except PermissionError:
            print(f"Permissioned denied: {file_path} exists, but cannot delete it")
    else:
        print(f"File {file_path} does not exist, skipping...")


if __name__ == "__main__":
    # files to remove from global
    # de,fr,en

    GLOBAL_DIR = "./most-common-words-multilingual/data/wordfrequency.info"
    OUTPUT = "concat.txt"
    FR = Path("./francais.txt")
    DE = Path("./wordlist-german.txt")

    print("=== DELETING FRENCH, GERMAN, ENGLISH FILES...")

    global_de = Path(GLOBAL_DIR) / "de.txt"
    global_fr = Path(GLOBAL_DIR) / "fr.txt"
    global_en = Path(GLOBAL_DIR) / "en.txt"

    delete_file(global_de)
    delete_file(global_fr)
    delete_file(global_en)

    print("=== MERGING GLOBAL FREQ FILES...")

    with Path(OUTPUT).open(mode="w", encoding="utf-8") as out:
        for data in Path(GLOBAL_DIR).iterdir():
            if data.is_file() and data.suffix == ".txt":
                try:
                    with data.open(mode="r", encoding="utf-8") as s:
                        for line in s:
                            out.write(line)
                except (OSError, IOError) as e:
                    print(f"Error reading {data}: {e}")

    print(f"=== PRUNING DUPLICATES IN {OUTPUT}...")
    temp_output = OUTPUT + ".tmp"
    remove_duplicates_system_sort(OUTPUT, temp_output)
    Path(temp_output).replace(Path(OUTPUT))

    print(f"=== REMOVING OVERLAPPING IN {OUTPUT}")
    remove_blacklisted_from_output(
        OUTPUT, "francais.txt", "wordlist-german.txt", out_file="output_clean.txt"
    )

    print(f"=== All done, global dataset in {OUTPUT}!")
