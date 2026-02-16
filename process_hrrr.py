import os
import json
import boto3
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib
from datetime import datetime, timedelta, timezone
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.colors as mcolors

# --- Configuration ---
BUCKET_NAME = 'noaa-hrrr-bdp-pds'
OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_cycle():
    """Finds the latest available complete run on AWS S3."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    now = datetime.now(timezone.utc)
    
    # Check the last 3 hours to find a complete upload
    for i in range(3):
        cycle_time = now - timedelta(hours=i)
        cycle_hour = cycle_time.hour
        date_str = cycle_time.strftime('%Y%m%d')
        
        # Check for the existence of the first subhourly file
        prefix = f"hrrr.{date_str}/conus/hrrr.t{cycle_hour:02d}z.wrfsubhf00.grib2"
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=prefix)
            return cycle_time.replace(minute=0, second=0, microsecond=0)
        except:
            continue
    return None

def download_file(cycle_dt, fhour):
    """Downloads a specific forecast hour file."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    date_str = cycle_dt.strftime('%Y%m%d')
    cycle_hour = cycle_dt.hour
    
    filename = f"hrrr.t{cycle_hour:02d}z.wrfsubhf{fhour:02d}.grib2"
    key = f"hrrr.{date_str}/conus/{filename}"
    local_path = f"temp_{filename}"
    
    print(f"Downloading {key}...")
    try:
        s3.download_file(BUCKET_NAME, key, local_path)
        return local_path
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return None

def process_grib(file_path, cycle_dt):
    """Extracts precip type and generates a transparent PNG."""
    grbs = pygrib.open(file_path)
    
    # HRRR Sub-hourly files often contain multiple time steps (e.g., 15, 30, 45, 60 min)
    # We group messages by validDate
    messages_by_time = {}
    
    try:
        # Filter for surface categorical variables
        relevant_vars = ['Categorical rain', 'Categorical freezing rain', 'Categorical ice pellets', 'Categorical snow']
        
        for g in grbs:
            if g.name in relevant_vars:
                if g.validDate not in messages_by_time:
                    messages_by_time[g.validDate] = {}
                messages_by_time[g.validDate][g.name] = g

    except Exception as e:
        print(f"Error reading GRIB: {e}")
        return []

    generated_frames = []

    for valid_time, msgs in sorted(messages_by_time.items()):
        if len(msgs) < 4:
            continue # Skip if missing data

        # Extract data grids
        crain = msgs['Categorical rain'].values
        cfrzr = msgs['Categorical freezing rain'].values
        cicep = msgs['Categorical ice pellets'].values
        csnow = msgs['Categorical snow'].values
        
        lats, lons = msgs['Categorical rain'].latlons()

        # Create a single "Precip Type" mask
        # 0=None, 1=Rain, 2=Freezing Rain, 3=Ice Pellets, 4=Snow
        ptype = np.zeros_like(crain)
        ptype = np.where(crain == 1, 1, ptype)
        ptype = np.where(cfrzr == 1, 2, ptype)
        ptype = np.where(cicep == 1, 3, ptype)
        ptype = np.where(csnow == 1, 4, ptype)
        
        # Mask out 0 (no precip) for transparency
        ptype_masked = np.ma.masked_where(ptype == 0, ptype)

        # Plotting
        fig = plt.figure(figsize=(10, 10), frameon=False)
        
        # Use a specific projection matching the data or standard PlateCarree for simple overlay
        # HRRR is Lambert Conformal, but for a simple overlay, we project to the map's view
        # However, transforming dense grids is slow. 
        # Fast hack: Plot using Cartopy to get the projection right, then save with no border.
        
        ax = plt.axes(projection=ccrs.Mercator())
        
        # Define bounds (CONUS zoom roughly)
        extent = [-125, -66.5, 24, 49.5]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Custom colormap: [Rain=Green, FrzRain=Pink, Ice=Orange, Snow=Blue]
        cmap = mcolors.ListedColormap(['#00ff00', '#ff00ff', '#ffa500', '#00ffff'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Reprojecting raster is heavy. We plot using pcolormesh with transform.
        # Downsample slightly for speed if needed, but quality was requested.
        mesh = ax.pcolormesh(lons, lats, ptype_masked, transform=ccrs.PlateCarree(), 
                             cmap=cmap, norm=norm, shading='nearest', alpha=0.8)

        ax.axis('off')
        
        # Save filename based on timestamp
        timestamp_str = valid_time.strftime("%Y%m%d_%H%M")
        out_filename = f"ptype_{timestamp_str}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=150)
        plt.close()

        # Metadata for this frame
        generated_frames.append({
            "file": f"data/{out_filename}",
            "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
            # Hardcoded bounds of the specific set_extent used above (in Lat/Lon)
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
    
    # Loop through forecast hours (0 to 18)
    # Note: 18 forecast hours is a lot of data. 
    # To keep GH Actions within time limits, we might do every 3rd hour or limit the range.
    # For now, we try 0-18.
    for fhour in range(0, 19): 
        local_file = download_file(cycle_dt, fhour)
        if local_file:
            frames = process_grib(local_file, cycle_dt)
            all_frames.extend(frames)
            os.remove(local_file) # Clean up space
            
    # Save metadata
    with open('site/metadata.json', 'w') as f:
        json.dump({"frames": all_frames, "cycle": cycle_dt.strftime("%Y-%m-%d %H:00 UTC")}, f)

if __name__ == "__main__":
    main()
