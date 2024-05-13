"""
$ ai-models --field fourcastnetv2_small
Grid: [0.25, 0.25]
Area: [90, 0, -90, 359.75]
Pressure levels:
   Levels: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
   Params: ['t', 'u', 'v', 'z', 'r', 'q']
Single levels:
   Params: ['10u', '10v', '2t', 'sp', 'msl', 'tcwv', '100u', '100v']
"""
import os, sys
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone

def get_latest_fcst_date():
    _timelist = []
    curr_time = datetime.now(timezone.utc)
    latest_fcst_time = curr_time.strftime('%Y%m%d')
    if curr_time.hour == 3 and curr_time.minute >= 30:
        latest_fcst_time += '00'
    elif curr_time.hour == 9 and curr_time.minute >= 30:
        latest_fcst_time += '06'
    elif curr_time.hour == 15 and curr_time.minute >= 30:
        latest_fcst_time += '12'
    elif curr_time.hour == 21 and curr_time.minute >= 30:
        latest_fcst_time += '18'
    elif 0 <= curr_time.hour < 3 or (curr_time.hour == 3 and curr_time.minute < 30):
        # overwrite
        latest_fcst_time = (curr_time - timedelta(days=1)).strftime('%Y%m%d')
        latest_fcst_time += '18'
    elif 3 < curr_time.hour < 9 or (curr_time.hour == 9 and curr_time.minute < 30):
        latest_fcst_time += '00'
    elif 9 < curr_time.hour < 15 or (curr_time.hour == 15 and curr_time.minute < 30):
        latest_fcst_time += '06'
    elif 15 < curr_time.hour < 21 or (curr_time.hour == 21 and curr_time.minute < 30):
        latest_fcst_time += '12'
    else:
        latest_fcst_time += '18'
    _timelist.append(latest_fcst_time)
    _prev_time = [datetime.strptime(latest_fcst_time, '%Y%m%d%H') - timedelta(hours=6*i) for i in range(1, 4)]
    _timelist += [_pt.strftime('%Y%m%d%H') for _pt in _prev_time]
    return _timelist

# download grib
latest = get_latest_fcst_date()
date, hour = latest[:-2], latest[-2:]
grib_url = f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{date}%2F{hour}%2Fatmos&file=gfs.t{hour}z.pgrb2.0p25.f000&var_HGT=on&var_MSLET=on&var_PWAT=on&var_PRES=on&var_RH=on&var_TMP=on&var_UGRD=on&var_VGRD=on&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_100_m_above_ground=on&lev_1000_mb=on&lev_925_mb=on&lev_850_mb=on&lev_700_mb=on&lev_600_mb=on&lev_500_mb=on&lev_400_mb=on&lev_300_mb=on&lev_250_mb=on&lev_200_mb=on&lev_150_mb=on&lev_100_mb=on&lev_50_mb=on&lev_surface=on&lev_mean_sea_level=on&lev_entire_atmosphere_(considered_as_a_single_layer)=on&subregion=&toplat=90&leftlon=0&rightlon=360&bottomlat=-90'
try:
    response = requests.get(grib_url, timeout=(30, 30))
except Exception:
    retry = 5
    retry_num = 0
    while retry_num <= retry:
        retry_num += 1
        if retry_num > retry:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [E] Cannot get grib data.")
            break
        try:
            print(f'Retrying {retry_num} time(s)...', end='')
            response = requests.get(grib_url, timeout=(30, 30))
        except Exception:
            print('failed')
        else:
            print('success')
            break
    if retry_num > retry:
        sys.exit(1)
if not response.status_code == 200:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [E] Cannot get grib data.")
    sys.exit(1)
with open('gfs.pgrb2.0p25.f000.grb', 'wb') as f:
    f.write(response.content)

# read grib file
gfs_file = 'gfs.pgrb2.0p25.f000.grb'
w10 = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
w100 = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 100}})
t2m = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
sp = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'surface'}})
msl = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'meanSea'}})
pwat = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'atmosphereSingleLayer'}})
upper = xr.open_dataset(gfs_file, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})

# save data as .npy file
data = np.empty((73, 721, 1440))
data[0] = w10['u10'][::-1,:].data.astype(np.float32)
data[1] = w10['v10'][::-1,:].data.astype(np.float32)
data[2] = w100['u100'][::-1,:].data.astype(np.float32)
data[3] = w100['v100'][::-1,:].data.astype(np.float32)
data[4] = t2m['t2m'][::-1,:].data.astype(np.float32)
data[5] = sp['sp'][::-1,:].data.astype(np.float32)
data[6] = msl['mslet'][::-1,:].data.astype(np.float32)
data[7] = pwat['pwat'][::-1,:].data.astype(np.float32)
data[8:21] = upper.variables['u'][::-1,::-1,:].data.astype(np.float32)
data[21:34] = upper.variables['v'][::-1,::-1,:].data.astype(np.float32)
data[34:47] = (upper.variables['gh'][::-1,::-1,:].data * 9.80665).astype(np.float32)
data[47:60] = upper.variables['t'][::-1,::-1,:].data.astype(np.float32)
data[60:] = upper.variables['r'][::-1,::-1,:].data.astype(np.float32)
np.save('input_gfs.npy', data)
