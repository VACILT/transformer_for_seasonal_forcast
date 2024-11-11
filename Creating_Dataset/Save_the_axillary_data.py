import numpy as np
import os
import h5py
import numpy.ma as ma


save_dir = "/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/NPZ_files/"
Fig_dir = '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Test/Figs/'

def seasonal_Winter_id(MONTH):
    win0= np.where(MONTH==11)[0]
    win1= np.where(MONTH==12)[0]
    win2= np.where(MONTH==1)[0]
    win3= np.where(MONTH==2)[0]
    win4= np.where(MONTH==3)[0]
    win5= np.where(MONTH==4)[0]
    win = np.concatenate(( win0, win1, win2, win3, win4, win5), axis=0)
    win = np.sort(win)
    return win

def seasonal_Winter_id_second(MONTH):
    win0= np.where(MONTH==11)[0]
    win1= np.where(MONTH==12)[0]
    win2= np.where(MONTH==1)[0]
    win3= np.where(MONTH==2)[0]
    win4= np.where(MONTH==3)[0]
    win = np.concatenate(( win0, win1, win2, win3, win4), axis=0)
    win = np.sort(win)
    return win


os.chdir(save_dir)

loaded = np.load('Zonal_mean_U_EP_Flux_Ens_00.npz')
Month = loaded['Month']
Win_id = seasonal_Winter_id(Month)
Win_id_4_NA = seasonal_Winter_id(Month[:5844])




U = np.zeros((4,6,5437,5,48,240))


for ensembel in range(6):
    if ensembel==0:
        with h5py.File(f"U_wind_Ens_{ensembel:02d}.h5", 'r') as hdf:
            lat = hdf['lat'][:]
            lon = hdf['lon'][:]
            plev = hdf['plev'][1:]
            U[0,ensembel,:,:,:,:] = hdf['U_C'][Win_id,1:,:,:]
            U[1,ensembel,:,:,:,:] = hdf['U_EA'][Win_id,1:,:,:]
            U[2,ensembel,:,:,:,:] = hdf['U_NA'][Win_id,1:,:,:]
            U[3,ensembel,:,:,:,:] = hdf['U_HI'][Win_id,1:,:,:]
    elif ensembel==4:
        with h5py.File(f"U_wind_Ens_{ensembel:02d}.h5", 'r') as hdf:
            U[0,ensembel,:,:,:,:] = hdf['U_C'][Win_id,1:,:,:]
            U[1,ensembel,:,:,:,:] = hdf['U_EA'][Win_id,1:,:,:]
            U[2,ensembel,:Win_id_4_NA.shape[0],:,:,:] = hdf['U_NA'][Win_id_4_NA,1:,:,:]
            U[3,ensembel,:,:,:,:] = hdf['U_HI'][Win_id,1:,:,:]
    else:
        with h5py.File(f"U_wind_Ens_{ensembel:02d}.h5", 'r') as hdf:
            U[0,ensembel,:,:,:,:] = hdf['U_C'][Win_id,1:,:,:]
            U[1,ensembel,:,:,:,:] = hdf['U_EA'][Win_id,1:,:,:]
            U[2,ensembel,:,:,:,:] = hdf['U_NA'][Win_id,1:,:,:]
            U[3,ensembel,:,:,:,:] = hdf['U_HI'][Win_id,1:,:,:]




#halfen the long resolution
U = ( U[:,:,:,:,:,0:240:2] + U[:,:,:,:,:,1:240:2] ) / 2




U = np.delete(U, -1, axis=4)
U = np.delete(U, 0, axis=4)

U_bool = np.zeros((U.shape)).astype(bool)
U_bool[2,4,Win_id_4_NA.shape[0]:,:,:,:] = True 
U_max = ma.masked_array(U,  mask=U_bool).max(axis = (0,1,2))
U_min = ma.masked_array(U,  mask=U_bool).min(axis = (0,1,2))

del U
del U_bool


lat = np.delete(lat, -1, axis=0)
lat = np.delete(lat, 0, axis=0)

lon = ( lon[0:240:2] + lon[1:240:2] ) / 2 

plev_u = plev



EP_V = ma.zeros(( 4, 6, 5437, 40, 48))
#EP_M = ma.zeros(( 4, 6, 4537, 40, 48))
EPD = ma.zeros(( 4, 6, 5437, 40, 48))
#Uh = ma.zeros(( 4, 6, 4537, 40, 48))

with h5py.File(f"EP_flux_all_ens_data.h5", 'r') as hdf:
    Plev = hdf['plev'][:]
    Lat = hdf['lat'][:]
    lat_id = np.where(Lat<20)[0].max()
    Lat = Lat[lat_id:]
    EP_V[0,:,:,:,:] = hdf['EP_V_C'][:,:,:,lat_id:]
    EP_V[1,:,:,:,:] = hdf['EP_V_EA'][:,:,:,lat_id:]
    EP_V[2,:,:,:,:] = hdf['EP_V_NA'][:,:,:,lat_id:]
    EP_V[3,:,:,:,:] = hdf['EP_V_HI'][:,:,:,lat_id:]
    EPD[0,:,:,:,:] = hdf['EPD_C'][:,:,:,lat_id:]
    EPD[1,:,:,:,:] = hdf['EPD_EA'][:,:,:,lat_id:]
    EPD[2,:,:,:,:] = hdf['EPD_NA'][:,:,:,lat_id:]
    EPD[3,:,:,:,:] = hdf['EPD_HI'][:,:,:,lat_id:]
    Month_data = hdf['Month'][:]
    nn_NA = hdf['nn_NA'][:]


#consider only 350 hPa-60 hPa

EP_V = EP_V[:,:,:,:12,:]   # consider only 350 hPa-60 (included) hPa      
EPD = EPD[:,:,:,9:21,:]    # consider only 100 hPa-1 (included) hPa      

plev_EPV = Plev[:12]
plev_EPD = Plev[9:21]

# removing the first and last lat
EP_V = np.delete(EP_V, -1, axis=4)
EP_V = np.delete(EP_V, 0, axis=4)

EPD = np.delete(EPD, -1, axis=4)
EPD = np.delete(EPD, 0, axis=4)

EP_mask = np.zeros((EPD.shape)).astype(bool)
EP_mask[2,4,int(nn_NA[4]):,:,:] = True 



EP_V_min = ma.masked_array(EP_V,  mask=EP_mask).min(axis = (0,1,2))
EP_V_max = ma.masked_array(EP_V,  mask=EP_mask).max(axis = (0,1,2))

EPD_min = ma.masked_array(EPD,  mask=EP_mask).min(axis = (0,1,2))
EPD_max = ma.masked_array(EPD,  mask=EP_mask).max(axis = (0,1,2))

del EP_mask
del EP_V
del EPD




os.chdir(save_dir)

np.savez_compressed('Transformer_axillary_data.npz', lat = lat, lon = lon, U_max = U_max, U_min = U_min, plev_u = plev_u, plev_EPV = plev_EPV, EP_V_min = EP_V_min, EP_V_max = EP_V_max, plev_EPD = plev_EPD, EPD_min = EPD_min, EPD_max = EPD_max)










