import os
import json
import boto3
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygrib
import warnings
from datetime import datetime, timedelta, timezone
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.colors as mcolors

# --- Configuration ---
BUCKET_NAME = 'noaa-hrrr-bdp-pds'
OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Suppress Boto3 and Matplotlib warnings
warnings.filterwarnings("ignore")

# Initialize S3 client once
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Fixed geographic bounds for CONUS
# [West, East, South, North]
EXTENT = [-125, -66.5, 24, 49.5]

def get_latest_cycle():
    """Finds the most recent completed HRRR run (checks for F18)."""
    now = datetime.now(timezone.utc)
    for i in range(2, 6): 
        cycle_time = now - timedelta(hours=i)
        cycle_hour = cycle_time.hour
        date_str = cycle_time.strftime('%Y%m%d')
        
        # Check for the 18th forecast hour to ensure the cycle is finished uploading
        prefix = f"hrrr.{date_str}/conus/hrrr.t{cycle_hour:02d}z.wrfsubhf18.grib2"
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=prefix)
            return cycle_time.replace(minute=0, second=0, microsecond=0)
        except:
            continue
    return None

def download_file(cycle_dt, fhour):
    date_str = cycle_dt.strftime('%Y%m%d')
    cycle_hour = cycle_dt.hour
    filename = f"hrrr.t{cycle_hour:02d}z.wrfsubhf{fhour:02d}.grib2"
    key = f"hrrr.{date_str}/conus/{filename}"
    local_path = f"temp_{filename}"
    try:
        s3.download_file(BUCKET_NAME, key, local_path)
        return local_path
    except:
        return None

def process_grib(file_path, force_save=False):
    try:
        grbs = pygrib.open(file_path)
    except:
        return []

    messages_by_time = {}
    # Variable names are lowercase in the sub-hourly GRIB2 files
    target_vars = ['crain', 'csnow', 'cicep', 'cfrzr', 'prate']

    for g in grbs:
        if g.shortName in target_vars:
            v_time = g.validDate
            if v_time not in messages_by_time:
                messages_by_time[v_time] = {}
            messages_by_time[v_time][g.shortName] = g

    generated_frames = []
    for valid_time, msgs in sorted(messages_by_time.items()):
        ref_msg = list(msgs.values())[0]
        lats, lons = ref_msg.latlons()
        
        # 1=Rain, 2=FrzRain, 3=Ice, 4=Snow
        crain = msgs['crain'].values if 'crain' in msgs else np.zeros(lats.shape)
        cfrzr = msgs['cfrzr'].values if 'cfrzr' in msgs else np.zeros(lats.shape)
        cicep = msgs['cicep'].values if 'cicep' in msgs else np.zeros(lats.shape)
        csnow = msgs['csnow'].values if 'csnow' in msgs else np.zeros(lats.shape)
        
        prate = msgs['prate'].values if 'prate' in msgs else np.zeros(lats.shape)
        rate_mmhr = prate * 3600  # Convert kg/m²/s to mm/hr
        
        ptype = np.zeros_like(crain)
        ptype = np.where(crain > 0, 1, ptype)
        ptype = np.where(cfrzr > 0, 2, ptype)
        ptype = np.where(cicep > 0, 3, ptype)
        ptype = np.where(csnow > 0, 4, ptype)
        
        # Check for any precipitation
        max_rate = np.max(rate_mmhr)
        
        # Skip image generation for clear weather unless it's the first frame
        if max_rate <= 0 and not force_save:
            generated_frames.append({
                "file": None,
                "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
                "bounds": [[EXTENT[2], EXTENT[0]], [EXTENT[3], EXTENT[1]]]
            })
            continue

        # Plotting - High Resolution & Fixed Aspect
        fig = plt.figure(figsize=(15, 9), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.Mercator(), frameon=False)
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        
        # Define colormaps and norm for intensity
        norm = mcolors.LogNorm(vmin=0.01, vmax=200)
        type_configs = [
            (1, 'Greens'),   # Rain
            (2, 'RdPu'),     # FrzRain
            (3, 'Oranges'),  # Ice
            (4, 'Blues')     # Snow
        ]
        
        for type_val, cmap_name in type_configs:
            mask = (ptype == type_val) & (rate_mmhr > 0)
            if np.any(mask):
                masked_rate = np.ma.masked_where(~mask, rate_mmhr)
                ax.pcolormesh(lons, lats, masked_rate, 
                              transform=ccrs.PlateCarree(), cmap=cmap_name, norm=norm, 
                              shading='auto', antialiased=True)

        ax.axis('off')
        
        out_name = f"ptype_{valid_time.strftime('%Y%m%d_%H%M')}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Save without cropping (fixes jumping alignment), higher DPI for quality
        plt.savefig(out_path, transparent=True, dpi=200)
        plt.close()

        generated_frames.append({
            "file": f"data/{out_name}",
            "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
            "bounds": [[EXTENT[2], EXTENT[0]], [EXTENT[3], EXTENT[1]]]
        })
        
    return generated_frames

def main():
    cycle_dt = get_latest_cycle()
    if not cycle_dt:
        print("Waiting for latest HRRR run to complete...")
        return

    print(f"Starting Cycle: {cycle_dt}")
    all_frames = []
    
    for fhour in range(0, 19): 
        print(f"  -> Processing Hour {fhour}...")
        local_file = download_file(cycle_dt, fhour)
        if local_file:
            # Force first frame save to verify output even if clear
            all_frames.extend(process_grib(local_file, force_save=(fhour==0)))
            os.remove(local_file)
            
    with open('site/metadata.json', 'w') as f:
        json.dump({"frames": all_frames, "cycle": cycle_dt.strftime("%Y-%m-%d %H:00 UTC")}, f)
    
    print(f"Finished. Total frames: {len(all_frames)}")

if __name__ == "__main__":
    main()
