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

# Suppress warnings to keep logs clean
warnings.filterwarnings("ignore")

# Initialize S3 client ONCE globally
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def get_latest_cycle():
    """Finds the latest available complete run on AWS S3."""
    now = datetime.now(timezone.utc)
    
    # Check the last 3 hours
    for i in range(3):
        cycle_time = now - timedelta(hours=i)
        cycle_hour = cycle_time.hour
        date_str = cycle_time.strftime('%Y%m%d')
        
        # Check for existence of the first subhourly file
        prefix = f"hrrr.{date_str}/conus/hrrr.t{cycle_hour:02d}z.wrfsubhf00.grib2"
        try:
            print(f"Checking for cycle: {cycle_time}...")
            s3.head_object(Bucket=BUCKET_NAME, Key=prefix)
            return cycle_time.replace(minute=0, second=0, microsecond=0)
        except:
            continue
    return None

def download_file(cycle_dt, fhour):
    """Downloads a specific forecast hour file."""
    date_str = cycle_dt.strftime('%Y%m%d')
    cycle_hour = cycle_dt.hour
    
    filename = f"hrrr.t{cycle_hour:02d}z.wrfsubhf{fhour:02d}.grib2"
    key = f"hrrr.{date_str}/conus/{filename}"
    local_path = f"temp_{filename}"
    
    try:
        s3.download_file(BUCKET_NAME, key, local_path)
        return local_path
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return None

def process_grib(file_path):
    try:
        grbs = pygrib.open(file_path)
    except Exception as e:
        print(f"Could not open GRIB file: {e}")
        return []

    messages_by_time = {}
    
    # Let's see what's actually inside the file if we find nothing
    all_names = set()

    for g in grbs:
        all_names.add(g.shortName)
        # HRRR sometimes uses 'crain' (lowercase) or specific parameter numbers
        # We check for the standard shortNames used in the sub-hourly files
        if g.shortName in ['CRAIN', 'CSNOW', 'CICEP', 'CFRZR']:
            v_time = g.validDate
            if v_time not in messages_by_time:
                messages_by_time[v_time] = {}
            messages_by_time[v_time][g.shortName] = g

    if not messages_by_time:
        print(f"  !!! No precip variables found in {file_path}")
        print(f"  Available shortNames: {list(all_names)[:20]}...") # Print first 20 to debug
        return []

    generated_frames = []

    for valid_time, msgs in sorted(messages_by_time.items()):
        # Try to proceed even if one type is missing by creating a zero-array
        ref_msg = list(msgs.values())[0]
        lats, lons = ref_msg.latlons()
        shape = ref_msg.values.shape

        crain = msgs['CRAIN'].values if 'CRAIN' in msgs else np.zeros(shape)
        cfrzr = msgs['CFRZR'].values if 'CFRZR' in msgs else np.zeros(shape)
        cicep = msgs['CICEP'].values if 'CICEP' in msgs else np.zeros(shape)
        csnow = msgs['CSNOW'].values if 'CSNOW' in msgs else np.zeros(shape)
        
        ptype = np.zeros_like(crain)
        ptype = np.where(crain > 0.5, 1, ptype)
        ptype = np.where(cfrzr > 0.5, 2, ptype)
        ptype = np.where(cicep > 0.5, 3, ptype)
        ptype = np.where(csnow > 0.5, 4, ptype)
        
        if np.max(ptype) == 0:
            print(f"    - No precip detected at {valid_time} (Dry Map)")
            # We will still generate the frame so the map isn't broken
            # but it will just be a fully transparent PNG.

        # Plotting
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.axes(projection=ccrs.Mercator())
        extent = [-125, -66.5, 24, 49.5]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        cmap = mcolors.ListedColormap(['#00ff00', '#ff00ff', '#ffa500', '#00ffff'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ptype_masked = np.ma.masked_where(ptype == 0, ptype)
        
        ax.pcolormesh(lons, lats, ptype_masked, transform=ccrs.PlateCarree(), 
                      cmap=cmap, norm=norm, shading='nearest', alpha=0.8)

        ax.axis('off')
        
        timestamp_str = valid_time.strftime("%Y%m%d_%H%M")
        out_filename = f"ptype_{timestamp_str}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        # Lower DPI to speed up GitHub Actions
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=80)
        plt.close()

        generated_frames.append({
            "file": f"data/{out_filename}",
            "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
            "bounds": [[24, -125], [49.5, -66.5]]
        })
        
    return generated_frames

def main():
    cycle_dt = get_latest_cycle()
    if not cycle_dt:
        print("No suitable cycle found.")
        return

    print(f"Processing Cycle: {cycle_dt}")
    
    all_frames = []
    
    # Process hours 0 through 18
    for fhour in range(0, 19): 
        print(f"Downloading Hour {fhour}...")
        local_file = download_file(cycle_dt, fhour)
        if local_file:
            frames = process_grib(local_file)
            all_frames.extend(frames)
            os.remove(local_file)
            
    # Save metadata
    with open('site/metadata.json', 'w') as f:
        json.dump({"frames": all_frames, "cycle": cycle_dt.strftime("%Y-%m-%d %H:00 UTC")}, f)
    
    print(f"Done. Generated {len(all_frames)} frames.")

if __name__ == "__main__":
    main()
