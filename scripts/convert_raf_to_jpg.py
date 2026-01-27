import rawpy
import imageio.v2 as imageio
from pathlib import Path

PHOTO_DIR = Path("test")

for raf_path in list(PHOTO_DIR.glob("*.RAF")) + list(PHOTO_DIR.glob("*.raf")):
    try:
        with rawpy.imread(str(raf_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8
            )

        out_path = PHOTO_DIR / f"{raf_path.stem}.jpg"

        if out_path.exists():
            print(f"Skipping {raf_path.name} (JPG already exists)")
            continue

        imageio.imwrite(out_path, rgb, quality=95)
        print(f"Converted: {raf_path.name} → {out_path.name}")

    except Exception as e:
        print(f"Failed: {raf_path.name} → {e}")
