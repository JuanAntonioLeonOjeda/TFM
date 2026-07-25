import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from df_import import load_lumos, read_image


def check_environment():
    print("=" * 60)
    print("1. ENVIRONMENT")
    print("=" * 60)

    import sys
    print(f"Python : {sys.version.split()[0]}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    for name in ["pandas", "pydicom", "cv2", "sklearn"]:
        try:
            mod = __import__(name)
            print(f"{name:8s}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"{name:8s}: NOT INSTALLED")


def check_data():
    print("\n" + "=" * 60)
    print("2. DATASET")
    print("=" * 60)

    df = load_lumos()

    print(f"Rows (images)  : {len(df)}          (expected ~1620)")
    print(f"Unique patients: {df['patient_id'].nunique()}   (expected 803)")
    print(f"Columns        : {list(df.columns)}")

    print("\nClass distribution:")
    counts = df["label"].value_counts().sort_index()
    names = {0: "Normal", 1: "Osteopenia", 2: "Osteoporosis"}
    for label, n in counts.items():
        print(f"  {label} - {names.get(label, '?'):13s}: {n:4d}  ({n/len(df)*100:5.1f}%)")

    print("\nImages per patient:")
    print(df.groupby("patient_id").size().value_counts().sort_index().to_string())

    print("\nMissing values in key columns:")
    keys = [c for c in ["label", "bmd", "bmd_L1-L4", "t_value", "T_L1-L4", "age", "gender"]
            if c in df.columns]
    print(df[keys].isna().sum().to_string())

    print("\nRegression target (bmd):")
    print(df["bmd"].describe().to_string())

    return df


def check_images(df, n=4):
    print("\n" + "=" * 60)
    print("3. IMAGES")
    print("=" * 60)

    sample = df.sample(n, random_state=42)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    names = {0: "Normal", 1: "Osteopenia", 2: "Osteoporosis"}

    for ax, (_, row) in zip(np.atleast_1d(axes), sample.iterrows()):
        img = read_image(row["image_path"])
        print(f"patient {row['patient_id']:3d} | shape {str(img.shape):12s} | "
              f"range [{img.min():.2f}, {img.max():.2f}] | label {row['label']}")
        ax.imshow(img, cmap="gray")
        ax.set_title(f"ID {row['patient_id']} - {names.get(row['label'], '?')}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=120, bbox_inches="tight")
    print("\nSaved preview to: sanity_check.png")


if __name__ == "__main__":
    check_environment()
    df = check_data()
    check_images(df)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)