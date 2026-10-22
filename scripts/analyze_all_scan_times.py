import h5py
import datetime
import numpy as np

raw_files = [
    "data/insat3ds/raw/3SIMG_20OCT2026_0000_L1B_STD_V01R00.h5",
    "data/insat3ds/raw/3SIMG_20OCT2026_0030_L1B_STD_V01R00.h5",
    "data/insat3ds/raw/3SIMG_20OCT2026_0100_L1B_STD_V01R00.h5",
    "data/insat3ds/raw/3SIMG_20OCT2026_0130_L1B_STD_V01R00.h5",
    "data/insat3ds/raw/3SIMG_20OCT2026_0200_L1B_STD_V01R00.h5",
    "data/insat3ds/raw/3SIMG_20OCT2026_0230_L1B_STD_V01R00.h5",
    "data/insat3ds/raw/3SIMG_20OCT2026_0300_L1B_STD_V01R00.h5",
]

print("=" * 80)
print("EXHAUSTIVE SCAN-LINE TIMING ANALYSIS ACROSS ALL 7 FILES")
print("=" * 80)

all_file_times = []

for idx, fpath in enumerate(raw_files):
    with h5py.File(fpath, "r") as h5:
        slt = h5["SCAN_LINE_TIME"][:]
        slt_strs = [x.decode("utf-8").strip() for x in slt]
        
        # Parse datetimes with microsecond precision: format '20-OCT-2026 00:00:26.414'
        dts = []
        for s in slt_strs:
            # format has date, space, time with milliseconds
            dt = datetime.datetime.strptime(s, "%d-%b-%Y %H:%M:%S.%f")
            dts.append(dt)
        all_file_times.append(dts)
        
        # Intra-file timing:
        # Check sweep differences within the file
        sweep_diffs_sec = np.array([(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1)])
        
        # Check Indian regional bounds:
        # Row 492 to 1271 in 4km grid -> sweep 246 to 635
        # (492 // 2 = 246, 1271 // 2 = 635)
        regional_start = dts[246]
        regional_end = dts[635]
        regional_duration_sec = (regional_end - regional_start).total_seconds()
        
        total_duration_sec = (dts[-1] - dts[0]).total_seconds()
        
        print(f"File [{idx}] {h5.attrs.get('Unique_Id').decode()}:")
        print(f"  Total Sweeps: {len(dts)}")
        print(f"  Sweep 0 (Row 0-1, North Limb):    {slt_strs[0]}")
        print(f"  Sweep 246 (Row 492-493, 37N):     {slt_strs[246]}")
        print(f"  Sweep 635 (Row 1270-1271, 5N):    {slt_strs[635]}")
        print(f"  Sweep 1408 (Row 2814-2815, South):{slt_strs[-1]}")
        print(f"  Total Duration: {total_duration_sec:.3f} s ({total_duration_sec/60:.2f} min)")
        print(f"  Regional Duration: {regional_duration_sec:.3f} s ({regional_duration_sec/60:.2f} min)")
        print(f"  Sweep Delta: mean={np.mean(sweep_diffs_sec):.3f} s, min={np.min(sweep_diffs_sec):.3f} s, max={np.max(sweep_diffs_sec):.3f} s, std={np.std(sweep_diffs_sec):.3f} s")

# Inter-file consecutive comparisons
print("\n" + "=" * 80)
print("CONSECUTIVE SCAN ROW-BY-ROW RECURRENCE INTERVALS (SWEEP BY SWEEP)")
print("=" * 80)

for f_idx in range(len(all_file_times) - 1):
    dts_curr = all_file_times[f_idx]
    dts_next = all_file_times[f_idx + 1]
    
    # Calculate interval for every sweep
    intervals_sec = np.array([(dts_next[i] - dts_curr[i]).total_seconds() for i in range(len(dts_curr))])
    intervals_min = intervals_sec / 60.0
    
    # Regional crop sweeps (246 to 635)
    reg_intervals_sec = intervals_sec[246:636]
    reg_intervals_min = intervals_min[246:636]
    
    print(f"\nInterval: Scan {f_idx} -> Scan {f_idx+1}:")
    print(f"  Full Disk (1409 sweeps):")
    print(f"    Mean interval:   {np.mean(intervals_min):.6f} min ({np.mean(intervals_sec):.3f} s)")
    print(f"    Min interval:    {np.min(intervals_min):.6f} min ({np.min(intervals_sec):.3f} s)")
    print(f"    Max interval:    {np.max(intervals_min):.6f} min ({np.max(intervals_sec):.3f} s)")
    print(f"    Std deviation:   {np.std(intervals_sec):.6f} s")
    print(f"  Regional Crop (sweeps 246 to 635):")
    print(f"    Mean interval:   {np.mean(reg_intervals_min):.6f} min ({np.mean(reg_intervals_sec):.3f} s)")
    print(f"    Min interval:    {np.min(reg_intervals_min):.6f} min ({np.min(reg_intervals_sec):.3f} s)")
    print(f"    Max interval:    {np.max(reg_intervals_min):.6f} min ({np.max(reg_intervals_sec):.3f} s)")
    print(f"    Std deviation:   {np.std(reg_intervals_sec):.6f} s")
