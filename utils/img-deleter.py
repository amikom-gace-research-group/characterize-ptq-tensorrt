import pathlib

for i in range(2500, 5001):
    img_path = f"{pathlib.Path.cwd()}/assets/bromelia/image_{i}.jpg"
    img = pathlib.Path(img_path)
    img.unlink()

    print(f"Deleted {img_path}")

print("All done!")

