import numpy as np
import os
import argparse
import h5py

parser = argparse.ArgumentParser(description='insert Ensembel number and Year.')

# Add arguments
parser.add_argument('ensembel_num', type=int, help='Ensembel number "0,1,2,3,4,5"')
args = parser.parse_args()
ensembel = args.ensembel_num 



save_dir = "/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/NPZ_files/"
Fig_dir = '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Test/Figs/'



Control_daily_dir = ['/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Control_run/Ensembel_01/Daily_data/Pressure_level/',
                     '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Control_run/Ensembel_02/Daily_data/Pressure_level/',
                     '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Control_run/Ensembel_03/Daily_data/Pressure_level/',
                     '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Control_run/Ensembel_04/Daily_data/Pressure_level/',
                     '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Control_run/Ensembel_05/Daily_data/Pressure_level/',
                     '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Control_run/Ensembel_06/Daily_data/Pressure_level/']
                     

Control_File_name_Daily_base = ["Pressure_level_data_00_",
                                "Pressure_level_data_01_",
                                "Pressure_level_data_02_",
                                "Pressure_level_data_03_",
                                "Pressure_level_data_04_",
                                "Pressure_level_data_05_"]





EA_daily_dir = ['/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/East_Asia/Ensembel_01/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/East_Asia/Ensembel_02/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/East_Asia/Ensembel_03/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/East_Asia/Ensembel_04/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/East_Asia/Ensembel_05/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/East_Asia/Ensembel_06/Daily_data/Pressure_level/']

EA_File_name_Daily_base = ["Pressure_level_data_EA_00_",
                           "Pressure_level_data_EA_01_",
                           "Pressure_level_data_EA_02_",
                           "Pressure_level_data_EA_03_",
                           "Pressure_level_data_EA_04_",
                           "Pressure_level_data_EA_05_"]






NA_daily_dir = ['/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/North_West_America/Ensembel_01/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/North_West_America/Ensembel_02/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/North_West_America/Ensembel_03/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/North_West_America/Ensembel_04/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/North_West_America/Ensembel_05/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/North_West_America/Ensembel_06/Daily_data/Pressure_level/']

NA_File_name_Daily_base = ["Pressure_level_data_NA_00_",
                           "Pressure_level_data_NA_01_",
                           "Pressure_level_data_NA_02_",
                           "Pressure_level_data_NA_03_",
                           "Pressure_level_data_NA_04_",
                           "Pressure_level_data_NA_05_"]





HI_daily_dir = ['/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Himalaya/Ensembel_01/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Himalaya/Ensembel_02/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Himalaya/Ensembel_03/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Himalaya/Ensembel_04/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Himalaya/Ensembel_05/Daily_data/Pressure_level/',
                '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Himalaya/Ensembel_06/Daily_data/Pressure_level/']

HI_File_name_Daily_base = ["Pressure_level_data_HI_00_",
                           "Pressure_level_data_HI_01_",
                           "Pressure_level_data_HI_02_",
                           "Pressure_level_data_HI_03_",
                           "Pressure_level_data_HI_04_",
                           "Pressure_level_data_HI_05_"]




npz_str = ".npz"




os.chdir(Control_daily_dir[ensembel])
I = 1990

fn = Control_File_name_Daily_base[ensembel] + str(I) + npz_str
loaded = np.load(fn)
lat = loaded['lat']
lon = loaded['lon']
plev = loaded['plev']


p_id = [np.where(plev ==85000)[0][0], np.where(plev ==50000)[0][0], np.where(plev ==20000)[0][0], np.where(plev ==6000)[0][0], np.where(plev ==1000)[0][0], np.where(plev ==100)[0][0]]
plev = plev[p_id]

lat_id = np.where(lat<20)[0].max()
lat = lat[lat_id:]
lat_size = 120-lat_id







U_C = np.zeros((10980,6,lat_size,240))   

data_id = 0
os.chdir(Control_daily_dir[ensembel])
for I in range(1990,2020):
    fn = Control_File_name_Daily_base[ensembel] + str(I) + npz_str
    loaded = np.load(fn)
    A = loaded['u_P']
    shape_data = A.shape[0]
    U_C[data_id:data_id+shape_data,:,:,:] = A[:,p_id,lat_id:,:]
    data_id += shape_data
    del A

U_C = U_C[:data_id,:,:,:]





U_EA = np.zeros((10980,6,lat_size,240))    

data_id = 0
os.chdir(EA_daily_dir[ensembel])
for I in range(1990,2020):
    fn = EA_File_name_Daily_base[ensembel] + str(I) + npz_str
    loaded = np.load(fn)
    A = loaded['u_P']
    shape_data = A.shape[0]
    U_EA[data_id:data_id+shape_data,:,:,:] = A[:,p_id,lat_id:,:]
    data_id += shape_data
    del A

U_EA = U_EA[:data_id,:,:,:]





U_NA = np.zeros((10980,6,lat_size,240)) 

if ensembel==4:
    data_id = 0
    os.chdir(NA_daily_dir[ensembel])
    for I in range(1990,2006):
        fn = NA_File_name_Daily_base[ensembel] + str(I) + npz_str
        loaded = np.load(fn)
        A = loaded['u_P']
        shape_data = A.shape[0]
        U_NA[data_id:data_id+shape_data,:,:,:] = A[:,p_id,lat_id:,:]
        data_id += shape_data
        del A
else:
    data_id = 0
    os.chdir(NA_daily_dir[ensembel])
    for I in range(1990,2020):
        fn = NA_File_name_Daily_base[ensembel] + str(I) + npz_str
        loaded = np.load(fn)
        A = loaded['u_P']
        shape_data = A.shape[0]
        U_NA[data_id:data_id+shape_data,:,:,:] = A[:,p_id,lat_id:,:]
        data_id += shape_data
        del A


U_NA = U_NA[:data_id,:,:,:]



U_HI = np.zeros((10980,6,lat_size,240))    
data_id = 0
os.chdir(HI_daily_dir[ensembel])
for I in range(1990,2020):
    fn = HI_File_name_Daily_base[ensembel] + str(I) + npz_str
    loaded = np.load(fn)
    A = loaded['u_P']
    shape_data = A.shape[0]
    U_HI[data_id:data_id+shape_data,:,:,:] = A[:,p_id,lat_id:,:]
    data_id += shape_data
    del A

U_HI = U_HI[:data_id,:,:,:]






os.chdir(save_dir)


def save_to_hdf5(save_path, lat, lon, plev, U_C, U_EA, U_NA, U_HI):
    with h5py.File(save_path, 'w') as hdf:
        hdf.create_dataset('lat', data=lat)
        hdf.create_dataset('lon', data=lon)
        hdf.create_dataset('plev', data=plev)
        hdf.create_dataset('U_C', data=U_C)
        hdf.create_dataset('U_EA', data=U_EA)
        hdf.create_dataset('U_NA', data=U_NA)
        hdf.create_dataset('U_HI', data=U_HI)
    print(f"Data saved to {save_path}")

save_path = os.path.join(save_dir, f"U_wind_Ens_{ensembel:02d}.h5")  # Change extension to .h5 for HDF5 format
save_to_hdf5(save_path, lat, lon, plev, U_C, U_EA, U_NA, U_HI)