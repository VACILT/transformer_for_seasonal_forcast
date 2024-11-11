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

for ensembel in range(1,6):
    os.chdir(save_dir)

    loaded = np.load('Zonal_mean_U_EP_Flux_Ens_00.npz')
    Month = loaded['Month']
    Year = loaded['Year']
    Days = loaded['Days']

    loaded = np.load('Transformer_axillary_data.npz')
    U_min = loaded['U_min']
    EP_V_min = loaded['EP_V_min']
    EPD_min = loaded['EPD_min']

    U_max = loaded['U_max']
    EP_V_max = loaded['EP_V_max']
    EPD_max = loaded['EPD_max']

    plev_u = loaded['plev_u']
    lat = loaded['lat']
    lon = loaded['lon']
    plev_EPD = loaded['plev_EPD']
    plev_EPV = loaded['plev_EPV']


    Win_id = seasonal_Winter_id(Month)
    Win_id_4_NA = seasonal_Winter_id(Month[:5844])

    U = np.zeros((4,5437,5,48,240))



    if ensembel==4:
        with h5py.File(f"U_wind_Ens_{ensembel:02d}.h5", 'r') as hdf:
            U[0,:,:,:,:] = hdf['U_C'][Win_id,1:,:,:]
            U[1,:,:,:,:] = hdf['U_EA'][Win_id,1:,:,:]
            U[2,:Win_id_4_NA.shape[0],:,:,:] = hdf['U_NA'][Win_id_4_NA,1:,:,:]
            U[3,:,:,:,:] = hdf['U_HI'][Win_id,1:,:,:]
    else:
        with h5py.File(f"U_wind_Ens_{ensembel:02d}.h5", 'r') as hdf:
            U[0,:,:,:,:] = hdf['U_C'][Win_id,1:,:,:]
            U[1,:,:,:,:] = hdf['U_EA'][Win_id,1:,:,:]
            U[2,:,:,:,:] = hdf['U_NA'][Win_id,1:,:,:]
            U[3,:,:,:,:] = hdf['U_HI'][Win_id,1:,:,:]




    #halfen the long resolution
    U = ( U[:,:,:,:,0:240:2] + U[:,:,:,:,1:240:2] ) / 2

    U = np.delete(U, -1, axis=3)
    U = np.delete(U, 0, axis=3)





    EP_V = ma.zeros(( 4, 5437, 40, 48))
    #EP_M = ma.zeros(( 4, 6, 4537, 40, 48))
    EPD = ma.zeros(( 4, 5437, 40, 48))
    #Uh = ma.zeros(( 4, 6, 4537, 40, 48))

    with h5py.File(f"EP_flux_all_ens_data.h5", 'r') as hdf:
        Plev = hdf['plev'][:]
        Lat = hdf['lat'][:]
        lat_id = np.where(Lat<20)[0].max()
        Lat = Lat[lat_id:]
        EP_V[0,:,:,:] = hdf['EP_V_C'][ensembel,:,:,lat_id:]
        EP_V[1,:,:,:] = hdf['EP_V_EA'][ensembel,:,:,lat_id:]
        EP_V[2,:,:,:] = hdf['EP_V_NA'][ensembel,:,:,lat_id:]
        EP_V[3,:,:,:] = hdf['EP_V_HI'][ensembel,:,:,lat_id:]
        EPD[0,:,:,:] = hdf['EPD_C'][ensembel,:,:,lat_id:]
        EPD[1,:,:,:] = hdf['EPD_EA'][ensembel,:,:,lat_id:]
        EPD[2,:,:,:] = hdf['EPD_NA'][ensembel,:,:,lat_id:]
        EPD[3,:,:,:] = hdf['EPD_HI'][ensembel,:,:,lat_id:]
        nn_NA = hdf['nn_NA'][:]


    #consider only 350 hPa-60 hPa

    EP_V = EP_V[:,:,:12,:]
    EPD = EPD[:,:,9:21,:]       # consider only 100 hPa-1 (included) hPa    




    # removing the first and last lat
    EP_V = np.delete(EP_V, -1, axis=3)
    EP_V = np.delete(EP_V, 0, axis=3)

    EPD = np.delete(EPD, -1, axis=3)
    EPD = np.delete(EPD, 0, axis=3)


    U = (U - U_min) / (U_max - U_min)
    EP_V = (EP_V - EP_V_min) / (EP_V_max - EP_V_min)
    EPD = (EPD - EPD_min) / (EPD_max - EPD_min)



    Month_AP = Month[Win_id]
    Year_AP = Year[Win_id]
    Days_AP = Days[Win_id]


    Winter_id_2 = seasonal_Winter_id_second(Month_AP)





    Winter_id_2 = Winter_id_2[:-18]



    Month_F = Month_AP[Winter_id_2]
    Year_F = Year_AP[Winter_id_2]
    Days_F = Days_AP[Winter_id_2]


    plev_u_out = plev_u[[1,3]]






    EP_V_in = np.zeros(( 4, Winter_id_2.shape[0], 7, 12, 46))
    EPD_in = np.zeros((4, Winter_id_2.shape[0], 7, 12, 46))
    U_in = np.zeros((4, Winter_id_2.shape[0], 5, 7, 46, 120))


    U_out = np.zeros(( 4, Winter_id_2.shape[0], 2, 7, 46, 120))


    for i in range(7):
        EP_V_in[:,:,i,:,:] = EP_V[:, Winter_id_2+i,:,:]
        EPD_in[:,:,i,:,:] = EPD[:, Winter_id_2+i,:,:]
        U_in[:,:,:,i,:,:] = U[:, Winter_id_2+i,:,:,:]
        U_out[:,:,0,i,:,:] = U[:, Winter_id_2+i+10,1,:,:]
        U_out[:,:,1,i,:,:] = U[:, Winter_id_2+i+10,3,:,:]


    del EP_V
    del EPD
    del U


    def flatten_first_dim(A, ensembel4=False, month_data=False):
        B = []
        for i in range(A.shape[0]):
            B.append(A[i])
        mm = B[2].shape[0]
        if ensembel4:
            mm = np.where(Month_F[Year_F<2006]==11)[0].max()
            B[2] = B[2][:mm]
        C = np.concatenate(( B[0], B[1], B[2], B[3]), axis=0)
        if month_data:
            M = np.concatenate(( Month_F, Month_F, Month_F[:mm], Month_F), axis=0)
            Y = np.concatenate(( Year_F, Year_F, Year_F[:mm], Year_F), axis=0)
            D = np.concatenate(( Days_F, Days_F, Days_F[:mm], Days_F), axis=0)
            E = np.concatenate(( np.zeros(Month_F.shape), np.ones(Month_F.shape), np.ones(Month_F[:mm].shape)*2, np.ones(Month_F.shape)*3), axis=0)
            return C, Y, M, D, E
        else:
            return C


    # Experiment_id, 0:C , 1:EA, 2:NA, 3,HI), the dates are the date of the starting date of the input time seri
    EP_V_in, Year_Data, Month_Data, Days_data, Experiment_id = flatten_first_dim(EP_V_in, ensembel4=(ensembel==4), month_data=True)
    EPD_in = flatten_first_dim(EPD_in, ensembel4=(ensembel==4), month_data=False)
    U_in = flatten_first_dim(U_in, ensembel4=(ensembel==4), month_data=False)
    U_out = flatten_first_dim(U_out, ensembel4=(ensembel==4), month_data=False)

    #cuting the lat
    lat = lat[1:]
    EP_V_in = EP_V_in[:,:,:,1:]
    EPD_in = EPD_in[:,:,:,1:]
    U_in = U_in[:,:,:,1:,:]
    U_out = U_out[:,:,:,1:,:]



    U_min = U_min[:,1:,:]
    EP_V_min = EP_V_min[:,1:]
    EPD_min = EPD_min[:,1:]

    U_max = U_max[:,1:,:]
    EP_V_max = EP_V_max[:,1:]
    EPD_max = EPD_max[:,1:]




    def making_the_patch_embeding(EP_V_in, EPD_in, U_in, U_out):
        EP_V_in = EP_V_in.reshape(EP_V_in.shape[0], 7, 12, 3, 15)
        # Shape: (18076, 7, 12, 3, 15)
        # Move the patch dimensions together
        EP_V_in = EP_V_in.transpose( 0, 3, 1, 2, 4)
        # Shape: (18076, 3, 7, 12, 15)
        EP_V_in = EP_V_in.reshape(EP_V_in.shape[0], 3, -1)
        # Shape: (18076, 3, 1260)
        EPD_in = EPD_in.reshape(EPD_in.shape[0], 7, 12, 3, 15)
        # Shape: (18076, 7, 12, 3, 15)
        EPD_in = EPD_in.transpose( 0, 3, 1, 2, 4)
        # Shape: (18076, 3, 7, 12, 15)
        EPD_in = EPD_in.reshape(EPD_in.shape[0], 3, -1)
        # Shape: (18076, 3, 1260)
        U_in = np.transpose(U_in, (0, 1, 2, 4, 3))
        # Reshape to split the dimensions (18076, 5, 7, 120, 45)
        U_in = U_in.reshape(U_in.shape[0],U_in.shape[1], 7, 10, 12, 3, 15)
        # Shape: (18076, 5, 7, 10, 12, 3, 15)
        U_in = U_in.transpose( 0, 1, 3, 5, 2, 4, 6)
        # Shape: (18076, 5, 10, 3, 7, 12, 15)
        U_in = U_in.reshape(U_in.shape[0],U_in.shape[1], 30, 7, 12, 15)
        # Shape: (18076, 5, 30, 7, 12, 15)
        U_in = U_in.reshape(U_in.shape[0],U_in.shape[1], 30, -1)
        # Shape: (18076, 5, 30, 1260)
        U_in = U_in.reshape(U_in.shape[0], 150, 1260)
        # Shape: (18076, 150, 1260)
        ###############
        U_out = np.transpose(U_out, (0, 1, 2, 4, 3))
        # Reshape to split the dimensions (18076, 2, 7, 120, 45)
        U_out = U_out.reshape(U_out.shape[0],U_out.shape[1], 7, 10, 12, 3, 15)
        # Shape: (18076, 2, 7, 10, 12, 3, 15)
        # Move the patch dimensions together
        U_out = U_out.transpose( 0, 1, 3, 5, 2, 4, 6)
        # Shape: (18076, 2, 10, 3, 7, 12, 15)
        U_out = U_out.reshape(U_out.shape[0],U_out.shape[1], 30, 7, 12, 15)
        # Shape: (18076, 2, 30, 7, 12, 15)
        U_out = U_out.reshape(U_out.shape[0],U_out.shape[1], 30, -1)
        # Shape: (18076, 2, 30, 1260)
        U_out = U_out.reshape(U_out.shape[0], 60, 1260)
        # Shape: (18076, 60, 1260)
        whole_data = np.zeros(( U_in.shape[0], 2, 156, 1260))
        whole_data[:,0,:150,:] = U_in
        whole_data[:,0,150:153,:] = EP_V_in
        whole_data[:,0,153:156,:] = EPD_in
        whole_data[:,1,:60,:] = U_out
        return whole_data


    whole_data = making_the_patch_embeding(EP_V_in, EPD_in, U_in, U_out)



    def inverse_patches_to_data(whole_data):
        U_in = whole_data[:,0,:150,:]
        EP_V_in = whole_data[:,0,150:153,:]
        EPD_in = whole_data[:,0,153:156,:]
        U_out = whole_data[:,1,:60,:] 
        U_in = U_in.reshape(U_in.shape[0], 5 ,10, 3, 7, 12, 15)
        # Shape: (18076, 5, 10, 3, 7, 12, 15)
        U_in = U_in.transpose(0,1,4,2,5,3,6)
        # Shape: (18076, 5, 7, 10, 12, 3, 15)
        U_in = U_in.reshape(U_in.shape[0], 5, 7, 120, 45)
        # Shape: (18076, 5, 7, 120, 45)
        U_out = U_out.reshape(U_out.shape[0], 2 ,10, 3, 7, 12, 15)
        # Shape: (18076, 2, 10, 3, 7, 12, 15)
        U_out = U_out.transpose(0,1,4,2,5,3,6)
        # Shape: (18076, 2, 7, 10, 12, 3, 15)
        U_out = U_out.reshape(U_out.shape[0], 2, 7, 120, 45)
        # Shape: (18076, 2, 7, 120, 45)
        EP_V_in = EP_V_in.reshape(EP_V_in.shape[0], 3, 7, 12, 15)
        # Shape: (18076, 3, 7, 12, 15)
        EP_V_in = EP_V_in.transpose(0,2,3,1,4)
        # Shape: (18076, 7, 12, 3, 15)
        EP_V_in = EP_V_in.reshape(EP_V_in.shape[0], 7, 12, 45)
        # Shape: (18076, 7, 12, 45)
        EPD_in = EPD_in.reshape(EPD_in.shape[0], 3, 7, 12, 15)
        # Shape: (18076, 3, 7, 12, 15)
        EPD_in = EPD_in.transpose(0,2,3,1,4)
        # Shape: (18076, 7, 12, 3, 15)
        EPD_in = EPD_in.reshape(EPD_in.shape[0], 7, 12, 45)
        # Shape: (18076, 7, 12, 45)
        return U_in, U_out, EP_V_in, EPD_in


    os.chdir(save_dir)
    def save_to_hdf5(save_path, whole_data):
        with h5py.File(save_path, 'w') as hdf:
            hdf.create_dataset('whole_data', data=whole_data)
        print(f"Data saved to {save_path}")


    save_path = os.path.join(save_dir, f"patches_data_{ensembel:02d}.h5")  # Change extension to .h5 for HDF5 format
    save_to_hdf5(save_path, whole_data)


    np.savez_compressed( f"Transformer_axillary_data_{ensembel:02d}.npz", lat = lat, lon = lon, U_max = U_max, U_min = U_min, plev_u = plev_u, plev_EPV = plev_EPV, EP_V_min = EP_V_min, EP_V_max = EP_V_max, plev_EPD = plev_EPD, EPD_min = EPD_min, EPD_max = EPD_max, plev_u_out = plev_u_out, Month_F = Month_F, Year_F = Year_F, Days_F = Days_F, Experiment_id = Experiment_id)