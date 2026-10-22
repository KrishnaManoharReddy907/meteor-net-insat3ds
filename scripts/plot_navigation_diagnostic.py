"""
scripts/plot_navigation_diagnostic.py
Generates a navigation diagnostic plot showing the INSAT-3DS full-disk coordinate grid
and the regional bounding box for the Indian subcontinent (5N-37N, 65E-97E).
Labeled explicitly as a navigation diagnostic, NOT a calibrated satellite image.
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RAW_FILE = "data/insat3ds/raw/3SIMG_20OCT2026_0000_L1B_STD_V01R00.h5"
OUT_FIG = "reports/figures/insat3ds_navigation_crop_diagnostic.png"


def generate_diagnostic_map():
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    
    with h5py.File(RAW_FILE, 'r') as h5:
        lat_raw = h5['Latitude'][:]
        lon_raw = h5['Longitude'][:]
        
        lat_scale = float(np.squeeze(h5['Latitude'].attrs.get('scale_factor', 0.01)))
        lat_offset = float(np.squeeze(h5['Latitude'].attrs.get('add_offset', 0.0)))
        lat_fill = int(np.squeeze(h5['Latitude'].attrs.get('_FillValue', 32767)))
        
        lon_scale = float(np.squeeze(h5['Longitude'].attrs.get('scale_factor', 0.01)))
        lon_offset = float(np.squeeze(h5['Longitude'].attrs.get('add_offset', 0.0)))
        lon_fill = int(np.squeeze(h5['Longitude'].attrs.get('_FillValue', 32767)))
        
        valid_geo = (lat_raw != lat_fill) & (lon_raw != lon_fill)
        
        lat_deg = np.full_like(lat_raw, np.nan, dtype=np.float32)
        lon_deg = np.full_like(lon_raw, np.nan, dtype=np.float32)
        lat_deg[valid_geo] = lat_raw[valid_geo] * lat_scale + lat_offset
        lon_deg[valid_geo] = lon_raw[valid_geo] * lon_scale + lon_offset
        
        # Target crop box: rows 492 to 1271, cols 943 to 1809
        row_min, row_max = 492, 1271
        col_min, col_max = 943, 1809
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        
        # Plot 1: Full-Disk Latitude Grid with Crop Box Overlay
        im1 = ax1.imshow(lat_deg, cmap='coolwarm', vmin=-60, vmax=60, origin='upper')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Latitude (degrees North)', fontsize=10)
        
        # Draw target crop bounding box on Full Disk
        rect = patches.Rectangle(
            (col_min, row_min), col_max - col_min, row_max - row_min,
            linewidth=2.0, edgecolor='lime', facecolor='none', linestyle='--',
            label=f'Target Crop Box ({row_max-row_min+1}x{col_max-col_min+1} px)\n[5N-37N, 65E-97E]'
        )
        ax1.add_patch(rect)
        
        # Mark Sub-Satellite Point (SSP: 1408, 1402)
        ax1.plot(1402, 1408, 'r+', markersize=12, markeredgewidth=2, label='SSP (0.0N, 82.0E)')
        
        # Mark key landmarks
        landmarks = {
            'Delhi': (1288, 669),
            'Mumbai': (1166, 898),
            'Chennai': (1355, 1052),
            'Kolkata': (1563, 811),
        }
        for name, (c, r) in landmarks.items():
            ax1.plot(c, r, 'yo', markersize=4)
            ax1.text(c + 20, r, name, color='yellow', fontsize=8, weight='bold')
            
        ax1.set_title('INSAT-3DS Full Disk: Decoded Latitude Grid & Target Crop\n(Navigation Diagnostic — Not Calibrated Imagery)', fontsize=11, weight='bold')
        ax1.set_xlabel('Column Index (GeoX: 0 to 2804)', fontsize=9)
        ax1.set_ylabel('Row Index (GeoY: 0 to 2815, North to South)', fontsize=9)
        ax1.legend(loc='lower left', fontsize=8, framealpha=0.85)
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        # Plot 2: Cropped Regional Latitude & Longitude Contours
        crop_lat = lat_deg[row_min:row_max+1, col_min:col_max+1]
        crop_lon = lon_deg[row_min:row_max+1, col_min:col_max+1]
        
        im2 = ax2.imshow(crop_lat, cmap='viridis', extent=[65, 97, 5, 37], origin='lower', aspect='auto')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Latitude (deg N)', fontsize=10)
        
        # Overlay Longitude and Latitude contours
        cs_lat = ax2.contour(np.linspace(65, 97, crop_lat.shape[1]), np.linspace(5, 37, crop_lat.shape[0]),
                             crop_lat, levels=np.arange(5, 40, 5), colors='white', linewidths=0.8, linestyles=':')
        ax2.clabel(cs_lat, inline=True, fontsize=7, fmt='%d°N')
        
        cs_lon = ax2.contour(np.linspace(65, 97, crop_lon.shape[1]), np.linspace(5, 37, crop_lon.shape[0]),
                             crop_lon, levels=np.arange(65, 100, 5), colors='cyan', linewidths=0.8, linestyles=':')
        ax2.clabel(cs_lon, inline=True, fontsize=7, fmt='%d°E')
        
        # Plot Landmark locations on crop
        crop_landmarks = {
            'New Delhi': (77.21, 28.61),
            'Mumbai': (72.88, 19.08),
            'Chennai': (80.27, 13.08),
            'Kolkata': (88.36, 22.57),
            'Kanyakumari': (77.55, 8.08),
            'Srinagar': (74.80, 34.08),
        }
        for name, (ln, lt) in crop_landmarks.items():
            ax2.plot(ln, lt, 'ro', markersize=5)
            ax2.text(ln + 0.5, lt - 0.2, name, color='black', fontsize=8, weight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            
        ax2.set_title(f'Target Regional Crop Domain (5°N–37°N, 65°E–97°E)\nGrid Dimensions: {row_max-row_min+1} rows × {col_max-col_min+1} cols (100% Valid Pixels)', fontsize=11, weight='bold')
        ax2.set_xlabel('Longitude (°E)', fontsize=9)
        ax2.set_ylabel('Latitude (°N)', fontsize=9)
        ax2.set_xlim(65, 97)
        ax2.set_ylim(5, 37)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUT_FIG, bbox_inches='tight')
        plt.close()
        
    print(f"Navigation diagnostic map generated successfully: {OUT_FIG}")


if __name__ == '__main__':
    generate_diagnostic_map()
