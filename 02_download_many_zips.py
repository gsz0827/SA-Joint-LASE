import os
from pathlib import Path

# ============================================================
# Use HuggingFace mirror
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"

from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# Config
# ============================================================

repo_id = "roslu/BBBC021-Human-MCF7-Cells"
repo_type = "dataset"

bbbc_root = Path(__file__).resolve().parent

out_dir = bbbc_root / "raw_download_all"
out_dir.mkdir(exist_ok=True)

MAX_ZIPS = 10


# ============================================================
# Main
# ============================================================

def main():
    print("Using HF endpoint:", os.environ.get("HF_ENDPOINT"))

    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])

    print("Listing repo files...")
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
    )

    zip_files = sorted([f for f in files if f.lower().endswith(".zip")])

    print("Found ZIP files:", len(zip_files))

    if len(zip_files) == 0:
        raise RuntimeError("No ZIP files found.")

    selected = zip_files[:MAX_ZIPS]

    print(f"\nDownloading {len(selected)} ZIP files:")
    for i, f in enumerate(selected):
        print(f"{i + 1:03d}: {f}")

    for i, target_file in enumerate(selected):
        print("\n" + "=" * 80)
        print(f"Downloading {i + 1}/{len(selected)}:")
        print(target_file)

        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=target_file,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )

        print("Downloaded to:")
        print(local_path)

    print("\nAll selected ZIP files downloaded.")
    print("Output dir:", out_dir.resolve())


if __name__ == "__main__":
    main()