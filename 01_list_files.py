from huggingface_hub import HfApi

repo_id = "roslu/BBBC021-Human-MCF7-Cells"
repo_type = "dataset"

api = HfApi()
files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

print("Total files:", len(files))

print("\nFirst 100 files:")
for f in files[:100]:
    print(f)

zip_files = [f for f in files if f.lower().endswith(".zip")]
img_files = [f for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]

print("\nNumber of ZIP files:", len(zip_files))
print("First 20 ZIP files:")
for f in zip_files[:20]:
    print(f)

print("\nNumber of image files:", len(img_files))
print("First 20 image files:")
for f in img_files[:20]:
    print(f)