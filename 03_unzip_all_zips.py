from pathlib import Path
import zipfile

bbbc_root = Path(__file__).resolve().parent

raw_dir = bbbc_root / "raw_download_all"
extract_dir = bbbc_root / "extracted_all_zips"
extract_dir.mkdir(exist_ok=True)

zip_files = sorted(raw_dir.rglob("*.zip"))

print("Found ZIP files:", len(zip_files))

if len(zip_files) == 0:
    raise RuntimeError("No ZIP files found in raw_download_all.")

for i, zip_path in enumerate(zip_files):
    print("\n" + "=" * 80)
    print(f"Extracting {i + 1}/{len(zip_files)}:")
    print(zip_path)

    # 每个 zip 解压到单独子目录，避免同名文件覆盖
    subdir = extract_dir / zip_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(subdir)

    print("Extracted to:")
    print(subdir)

print("\nAll ZIP files extracted.")
print("Output dir:", extract_dir.resolve())

all_files = list(extract_dir.rglob("*"))
print("Total extracted paths:", len(all_files))
print("\nFirst 50 extracted paths:")
for f in all_files[:50]:
    print(f)