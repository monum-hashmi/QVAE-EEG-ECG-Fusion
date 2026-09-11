import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def to_native(obj):
    """Convert MATLAB structs loaded by scipy.io.loadmat into Python dict/list structures."""
    if hasattr(obj, "_fieldnames"):
        return {field: to_native(getattr(obj, field)) for field in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.size == 1:
                return to_native(obj.item())
            return [to_native(x) for x in obj.flat]
        return obj
    return obj


def describe(obj, name="root", depth=0, max_depth=5):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{name}: ...")
        return

    if isinstance(obj, dict):
        print(f"{indent}{name}: dict with keys = {list(obj.keys())}")
        for k, v in obj.items():
            describe(v, k, depth + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{indent}{name}: list length = {len(obj)}")
        if len(obj) > 0:
            describe(obj[0], f"{name}[0]", depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{name}: ndarray shape={obj.shape}, dtype={obj.dtype}")
        if obj.size > 0 and obj.dtype != object and np.issubdtype(obj.dtype, np.number):
            flat = obj.ravel()
            print(f"{indent}  min={np.nanmin(flat):.4f}, max={np.nanmax(flat):.4f}, mean={np.nanmean(flat):.4f}")
    else:
        print(f"{indent}{name}: {type(obj).__name__} = {obj}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to DREAMER.mat")
    parser.add_argument("--max_depth", type=int, default=6)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    native = {k: to_native(v) for k, v in mat.items() if not k.startswith("__")}

    print("=" * 90)
    print(f"Loaded: {path}")
    print("=" * 90)
    describe(native, "MAT", max_depth=args.max_depth)


if __name__ == "__main__":
    main()
