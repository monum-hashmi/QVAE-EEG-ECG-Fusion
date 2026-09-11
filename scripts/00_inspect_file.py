import argparse
from pathlib import Path

def inspect_mat(path: Path):
    from scipy.io import loadmat
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    print(f"MAT file: {path}")
    print("=" * 80)
    for key, value in data.items():
        if key.startswith("__"):
            continue
        print(f"KEY: {key}")
        print(f"TYPE: {type(value)}")
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"SHAPE: {shape}")
        print(f"DTYPE: {dtype}")
        print("-" * 80)

def inspect_pickle_or_dat(path: Path):
    import pickle
    print(f"Pickle/DAT file: {path}")
    print("=" * 80)
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    print(f"TOP TYPE: {type(obj)}")
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"KEY: {key}")
            print(f"TYPE: {type(value)}")
            print(f"SHAPE: {getattr(value, 'shape', None)}")
            print(f"DTYPE: {getattr(value, 'dtype', None)}")
            print("-" * 80)
    else:
        print(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .mat, .dat, .pkl, .csv file")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".mat":
        inspect_mat(path)
    elif suffix in [".dat", ".pkl", ".pickle"]:
        inspect_pickle_or_dat(path)
    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        print(f"CSV file: {path}")
        print("=" * 80)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        print(df.head())
    else:
        print(f"Unsupported extension: {suffix}")
        print("Supported: .mat, .dat, .pkl, .pickle, .csv")

if __name__ == "__main__":
    main()
