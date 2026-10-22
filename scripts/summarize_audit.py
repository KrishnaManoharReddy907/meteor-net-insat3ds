import json

with open('reports/insat3ds_audit_summary.json') as f:
    data = json.load(f)

print("=== FILES SUMMARY ===")
for r in data['file_records']:
    print(f"{r['filename']} | {r['filesize_mb']:.2f} MB | Start: {r['start_datetime']} | End: {r['end_datetime']}")

print("\n=== CADENCE TABLE ===")
for row in data['cadence_table']:
    print(f"[{row['index']}] {row['filename']} | Start: {row['start_utc']} | Duration: {row['scan_duration_min']:.2f}m | Delta: {row['interval_from_prev_min']}m | Status: {row['status']}")

rec0 = data['file_records'][0]
print("\n=== DATASETS IN HDF5 ===")
for name, dinfo in rec0['datasets'].items():
    print(f"{name:20s} | shape: {str(dinfo['shape']):20s} | dtype: {dinfo['dtype']:8s}")

print("\n=== CHANNEL STATS (FIRST FILE) ===")
for ch, st in rec0['channel_stats'].items():
    print(f"{ch:10s} | shape: {st['shape']} | res: {st['resolution_km']} km | lambda: {st['central_wavelength']} um | fill: {st['fill_value']} | valid%: {st['valid_percentage']:.2f}% | DN min-max: {st['raw_min_dn']}-{st['raw_max_dn']} | mean: {st['raw_mean_dn']:.1f}")

print("\n=== LUT INFO (FIRST FILE) ===")
for lut, linfo in rec0['lut_info'].items():
    print(f"{lut:20s} | shape: {linfo['shape']} | units: {linfo['units']} | min-max: {linfo['min_value']} to {linfo['max_value']} | first 3: {linfo['first_5'][:3]} | last 3: {linfo['last_5'][-3:]}")

print("\n=== GEOLOCATION INFO ===")
for g, ginfo in rec0['geo_info'].items():
    vmin = ginfo.get('valid_min_deg', 'N/A')
    vmax = ginfo.get('valid_max_deg', 'N/A')
    print(f"{g:15s} | shape: {ginfo['shape']} | dtype: {ginfo['dtype']} | units: {ginfo['units']} | scale: {ginfo['scale_factor']} | fill: {ginfo['fill_value']} | valid range deg: {vmin} to {vmax}")

print("\n=== CHANNEL CONSISTENCY ACROSS ALL 7 FILES ===")
for r in data['file_records']:
    t1 = r['channel_stats']['IMG_TIR1']
    wv = r['channel_stats']['IMG_WV']
    print(f"{r['filename']} | TIR1 mean DN: {t1['raw_mean_dn']:.2f} (valid: {t1['valid_percentage']:.2f}%) | WV mean DN: {wv['raw_mean_dn']:.2f} (valid: {wv['valid_percentage']:.2f}%)")
