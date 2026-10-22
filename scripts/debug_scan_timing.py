import h5py
import numpy as np

fpath = "data/insat3ds/raw/3SIMG_20OCT2026_0000_L1B_STD_V01R00.h5"

with h5py.File(fpath, "r") as h5:
    slt = h5["SCAN_LINE_TIME"]
    geoy = h5["GeoY"]
    geoy2 = h5["GeoY2"]
    
    print("=== SCAN_LINE_TIME ===")
    print("Shape:", slt.shape, "Dtype:", slt.dtype)
    for k, v in slt.attrs.items():
        print(f"  Attr: {k} = {v}")
        
    print("\n=== GeoY2 ===")
    print("Shape:", geoy2.shape, "Dtype:", geoy2.dtype)
    for k, v in geoy2.attrs.items():
        print(f"  Attr: {k} = {v}")
    print("First 10 GeoY2 values:", geoy2[:10])
    print("Last 10 GeoY2 values:", geoy2[-10:])
    
    print("\n=== GeoY ===")
    print("Shape:", geoy.shape, "Dtype:", geoy.dtype)
    for k, v in geoy.attrs.items():
        print(f"  Attr: {k} = {v}")
    print("First 10 GeoY values:", geoy[:10])
    print("Last 10 GeoY values:", geoy[-10:])
