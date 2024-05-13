"""
Some codes are from ai-models module by ECMWF-lab
"""
import os

import numpy as np

import torch
import fourcastnetv2 as fcnv2

import netCDF4 as nc
from datetime import datetime, timedelta

def get_free_gpu_memory(gpu_index):
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    # Convert output to list of integers representing free memory for each GPU
    free_memory = [float(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
    return free_memory[gpu_index]

def normalise(data, stds, means, reverse=False):
    """Normalise data using pre-saved global statistics"""
    if reverse:
        new_data = data * stds + means
    else:
        new_data = (data - means) / stds
    return new_data

if __name__ == '__main__':
    # check GPU availability
    RUN_MODE = 'GPU'
    if torch.cuda.is_available():
        current_gpu_index = torch.cuda.current_device()
        free_memory = get_free_gpu_memory(current_gpu_index) / 1024
        if free_memory < 12.5:
            RUN_MODE = 'CPU'
    else:
        RUN_MODE = 'CPU'
    
    # set running fcst hours
    max_hour = 240
    fcst_hours = list(range(6, max_hour + 6, 6))
    
    # Load model
    device = 'cuda' if RUN_MODE == 'GPU' else 'cpu'
    
    means = np.load("global_means.npy")
    means = means[:, :73, ...].astype(np.float32)

    stds = np.load("global_stds.npy")
    stds = stds[:, :73, ...].astype(np.float32)
    
    input_data = np.load('input_gfs.npy').astype(np.float32)
    input_data = normalise(input_data, stds, means)
    input_data = torch.from_numpy(input_data).to(device)
    
    model = fcnv2.FourierNeuralOperatorNet().to(device)
    model.zero_grad()
    
    checkpoint = torch.load("weights.tar", map_location=device)
    weights = checkpoint["model_state"]
    drop_vars = ["module.norm.weight", "module.norm.bias"]
    weights = {k: v for k, v in weights.items() if k not in drop_vars}
    
    # Make sure the parameter names are the same as the checkpoint
    # need to use strict = False to avoid this error message when
    # using sfno_76ch::
    # RuntimeError: Error(s) in loading state_dict for Wrapper:
    # Missing key(s) in state_dict: "module.trans_down.weights",
    # "module.itrans_up.pct",
    try:
        # Try adding model weights as dictionary
        new_state_dict = dict()
        for k, v in checkpoint["model_state"].items():
            name = k[7:]
            if name != "ged":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(checkpoint["model_state"])
    
    # Set model to eval mode
    model.eval()
    
    # with torch.no_grad():
    torch.set_grad_enabled(False)
    
    # Run the inference session
    for i, fcst_hour in enumerate(fcst_hours): # run to 240 hrs in every 6hrs
        output_file = f'fcnv2_inference_gfs_{"%03d" % fcst_hour}.nc'
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [I] Running for FCNv2-GFS [+{fcst_hour}h]...")
        
        output = model(input_data)
        input_data = output
        
        output = normalise(output.cpu().numpy(), stds, means, reverse=True)
        output = output[0, ...]
        
        with nc.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_file:
            # Create dimensions
            nc_file.createDimension("level", 13)
            nc_file.createDimension("longitude", 1440)
            nc_file.createDimension("latitude", 721)
            
            # Create variables
            nc_lev = nc_file.createVariable("level", np.float32, ("level",))
            nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude",))
            nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude",))
            nc_u10 = nc_file.createVariable("u10", np.float32, ("latitude", "longitude"))
            nc_v10 = nc_file.createVariable("v10", np.float32, ("latitude", "longitude"))
            nc_u100 = nc_file.createVariable("u100", np.float32, ("latitude", "longitude"))
            nc_v100 = nc_file.createVariable("v100", np.float32, ("latitude", "longitude"))
            nc_t2m = nc_file.createVariable("t2m", np.float32, ("latitude", "longitude"))
            nc_sp = nc_file.createVariable("sp", np.float32, ("latitude", "longitude"))
            nc_msl = nc_file.createVariable("msl", np.float32, ("latitude", "longitude"))
            nc_tcwv = nc_file.createVariable("tcwv", np.float32, ("latitude", "longitude"))
            nc_u = nc_file.createVariable("u", np.float32, ("level", "latitude", "longitude"))
            nc_v = nc_file.createVariable("v", np.float32, ("level", "latitude", "longitude"))
            nc_z = nc_file.createVariable("z", np.float32, ("level", "latitude", "longitude"))
            nc_t = nc_file.createVariable("t", np.float32, ("level", "latitude", "longitude"))
            nc_r = nc_file.createVariable("r", np.float32, ("level", "latitude", "longitude"))
            
            # Set variable attributes
            nc_u10.units = "m/s"
            nc_v10.units = "m/s"
            nc_u100.units = "m/s"
            nc_v100.units = "m/s"
            nc_t2m.units = "K"
            nc_sp.units = "Pa"
            nc_msl.units = "Pa"
            nc_tcwv.units = "kg/m^2"
            nc_u.units = "m/s"
            nc_v.units = "m/s"
            nc_z.units = "m2/s2"
            nc_t.units = "K"
            nc_r.units = "%"
            
            # Write data to variables
            nc_lev[:] = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
            nc_lon[:] = np.linspace(0, 359.75, 1440)
            nc_lat[:] = np.linspace(90, -90, 721)
            nc_u10[:] = output[0]
            nc_v10[:] = output[1]
            nc_u100[:] = output[2]
            nc_v100[:] = output[3]
            nc_t2m[:] = output[4]
            nc_sp[:] = output[5]
            nc_msl[:] = output[6]
            nc_tcwv[:] = output[7]
            nc_u[:] = output[8:21]
            nc_v[:] = output[21:34]
            nc_z[:] = output[34:47]
            nc_t[:] = output[47:60]
            nc_r[:] = output[60:]
            
            # set attributes
            setattr(nc_file, 'description', 'FourCastNet v2-Small Model Inference Data')
            setattr(nc_file, 'reference', 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm')
            setattr(nc_file, 'init_field', 'NCEP-GFS analysis in 0.25 degree grid')
            setattr(nc_file, 'code_author', 'BigShuiTai')
            setattr(nc_file, 'time', runtime)
            setattr(nc_file, 'step', fcst_hour)
