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

warnings.filterwarnings("ignore")
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def get_latest_cycle():
    """Finds a cycle that is likely complete (at least 2 hours old)."""
    now = datetime.now(timezone.utc)
    
    # We look back starting from 2 hours ago to ensure the run is finished uploading
    for i in range(2, 6): 
        cycle_time = now - timedelta(hours=i)
        cycle_hour = cycle_time.hour
        date_str = cycle_time.strftime('%Y%m%d')
        
        # Check for hour 18 specifically to ensure the whole run is there
        prefix = f"hrrr.{date_str}/conus/hrrr.t{cycle_hour:02d}z.wrfsubhf18.grib2"
        try:
            print(f"Checking if cycle {date_str} T{cycle_hour}Z is complete...")
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
    except Exception as e:
        print(f"  X Skillpped {filename}: {e}")
        return None

def process_grib(file_path, force_save=False):
    try:
        grbs = pygrib.open(file_path)
    except:
        return []

    messages_by_time = {}
    target_vars = ['crain', 'csnow', 'cicep', 'cfrzr']

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
        
        crain = msgs['crain'].values if 'crain' in msgs else np.zeros(lats.shape)
        cfrzr = msgs['cfrzr'].values if 'cfrzr' in msgs else np.zeros(lats.shape)
        cicep = msgs['cicep'].values if 'cicep' in msgs else np.zeros(lats.shape)
        csnow = msgs['csnow'].values if 'csnow' in msgs else np.zeros(lats.shape)
        
        ptype = np.zeros_like(crain)
        ptype = np.where(crain > 0, 1, ptype) # Any value > 0
        ptype = np.where(cfrzr > 0, 2, ptype)
        ptype = np.where(cicep > 0, 3, ptype)
        ptype = np.where(csnow > 0, 4, ptype)
        
        max_val = np.max(ptype)
        
        # Only save if there is precip OR if we are forcing a test frame
        if max_val == 0 and not force_save:
            print(f"    - Clear skies at {valid_time}")
            generated_frames.append({
                "file": None,
                "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
                "bounds": [[24, -125], [49.5, -66.5]]
            })
            continue

        print(f"    + Saving frame for {valid_time} (Max P-Type: {max_val})")
        fig = plt.figure(figsize=(10, 6), frameon=False)
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
        
        cmap = mcolors.ListedColormap(['#00ff00', '#ff00ff', '#ffa500', '#00ffff'])
        norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
        
        ax.pcolormesh(lons, lats, np.ma.masked_where(ptype == 0, ptype), 
                      transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading='nearest')
        ax.axis('off')
        
        out_filename = f"ptype_{valid_time.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, out_filename), bbox_inches='tight', pad_inches=0, transparent=True, dpi=90)
        plt.close()

        generated_frames.append({
            "file": f"data/{out_filename}",
            "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
            "bounds": [[24, -125], [49.5, -66.5]]
        })
        
    return generated_frames

# Update main loop to pass force_save=True for the first file
def main():
    cycle_dt = get_latest_cycle()
    if not cycle_dt: return

    all_frames = []
    for fhour in range(0, 19): 
        local_file = download_file(cycle_dt, fhour)
        if local_file:
            # Force save the very first timestep of the first file
            is_first = (fhour == 0)
            all_frames.extend(process_grib(local_file, force_save=is_first))
            os.remove(local_file)
    # ... (rest of the script)

def main():
    cycle_dt = get_latest_cycle()
    if not cycle_dt:
        print("No complete cycle found yet.")
        return

    print(f"Processing Verified Cycle: {cycle_dt}")
    all_frames = []
    
    for fhour in range(0, 19): 
        print(f"Working on F{fhour:02d}...")
        local_file = download_file(cycle_dt, fhour)
        if local_file:
            all_frames.extend(process_grib(local_file))
            os.remove(local_file)
            
    with open('site/metadata.json', 'w') as f:
        json.dump({"frames": all_frames, "cycle": cycle_dt.strftime("%Y-%m-%d %H:00 UTC")}, f)
    
    print(f"Successfully processed {len(all_frames)} timesteps.")

if __name__ == "__main__":
    main()
