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

# Suppress warnings
warnings.filterwarnings("ignore")

# S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# CONUS bounds [West, East, South, North]
EXTENT = [-125, -66.5, 24, 49.5]

def get_latest_cycle():
    now = datetime.now(timezone.utc)
    for i in range(2, 6): 
        cycle_time = now - timedelta(hours=i)
        cycle_hour = cycle_time.hour
        date_str = cycle_time.strftime('%Y%m%d')
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
        
        crain = msgs['crain'].values if 'crain' in msgs else np.zeros(lats.shape)
        cfrzr = msgs['cfrzr'].values if 'cfrzr' in msgs else np.zeros(lats.shape)
        cicep = msgs['cicep'].values if 'cicep' in msgs else np.zeros(lats.shape)
        csnow = msgs['csnow'].values if 'csnow' in msgs else np.zeros(lats.shape)
        
        prate = msgs['prate'].values if 'prate' in msgs else np.zeros(lats.shape)
        rate_mmhr = prate * 3600  # kg/m²/s → mm/hr
        
        ptype = np.zeros_like(crain, dtype=int)
        ptype = np.where(crain > 0, 1, ptype)   # Rain
        ptype = np.where(cfrzr > 0, 2, ptype)   # Freezing Rain
        ptype = np.where(cicep > 0, 3, ptype)   # Ice Pellets
        ptype = np.where(csnow > 0, 4, ptype)   # Snow
        
        max_rate = np.max(rate_mmhr)
        
        if max_rate <= 0 and not force_save:
            generated_frames.append({
                "file": None,
                "time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
                "bounds": [[EXTENT[2], EXTENT[0]], [EXTENT[3], EXTENT[1]]]
            })
            continue

        fig = plt.figure(figsize=(15, 9), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.Mercator(), frameon=False)
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        
        # Intensity norm
        norm = mcolors.LogNorm(vmin=0.01, vmax=100)
        
        # Custom rain colormap: light green → dark red via yellow/orange
        rain_colors = [
            '#98FB98',  # pale green (very light)
            '#32CD32',  # lime green
            '#ADFF2F',  # greenyellow
            '#FFFF00',  # yellow
            '#FFD700',  # gold (transition)
            '#FFA500',  # orange
            '#FF4500',  # orangered
            '#DC143C',  # crimson
            '#8B0000'   # darkred
        ]
        rain_cmap = mcolors.LinearSegmentedColormap.from_list('custom_rain', rain_colors, N=256)

        type_configs = [
            (1, rain_cmap),     # Rain: custom
            (2, 'RdPu'),        # Freezing Rain: pink → magenta
            (3, 'Purples'),     # Ice Pellets: purple shades
            (4, 'Blues')        # Snow: light → dark blue
        ]
        
        for type_val, cmap in type_configs:
            mask = (ptype == type_val) & (rate_mmhr > 0.005)
            if np.any(mask):
                masked_rate = np.ma.masked_where(~mask, rate_mmhr)
                ax.pcolormesh(lons, lats, masked_rate, 
                              transform=ccrs.PlateCarree(),
                              cmap=cmap,
                              norm=norm,
                              shading='nearest',
                              antialiased=False,
                              zorder=5)

        ax.axis('off')
        
        out_name = f"ptype_{valid_time.strftime('%Y%m%d_%H%M')}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        plt.savefig(out_path, transparent=True, dpi=700, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

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
            all_frames.extend(process_grib(local_file, force_save=(fhour==0)))
            os.remove(local_file)
            
    with open('site/metadata.json', 'w') as f:
        json.dump({"frames": all_frames, "cycle": cycle_dt.strftime("%Y-%m-%d %H:00 UTC")}, f)
    
    print(f"Finished. Total frames: {len(all_frames)}")

if __name__ == "__main__":
    main()
