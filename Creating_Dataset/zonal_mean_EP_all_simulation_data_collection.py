import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import h5py




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




def filtering(data,limit):
    lower_bound = (-1)*limit
    upper_bound = limit
    mask_low = data < lower_bound
    data[mask_low] = lower_bound
    mask_high = data > upper_bound
    data[mask_high] = upper_bound
    return data



def plot_climatology(aa,aa1,cc1,cc2,bb):
    ax = plt.axes() 
    ax.set_aspect(9)
    levels = 13
    mmax=max([aa.max(),(-1)*(aa.min())])
    level_boundaries = np.linspace((-1)*mmax, mmax, levels + 1)
    p = ax.contourf( lat, Plev_lg1, aa,level_boundaries,cmap='seismic',vmin = (-1)*mmax, vmax = mmax)
    vecplot = ax.quiver( lat, Plev_lg1, cc1, cc2, width=0.0008)
    level_boundaries_U = np.round(np.arange(-100, 100, 10) ,0)
    cou_pt = ax.contour( lat, Plev_lg1, aa1, level_boundaries_U, colors='k',linewidths=0.8)
    ax.clabel(cou_pt, inline=True, fontsize=11 ,fmt=  '%1.0f')
    plt.title(bb)
    plt.yticks(y_positions, y_labels)
    ax.quiverkey(vecplot, X=.65, Y=.89, U=1E6, label=r'$1.0e06 \frac{m^3}{s^2}$', labelpos='E', coordinates='figure', color='k')
    plt.colorbar(p)
    plt.show()



limit_fz = 30000000
ep_lim = 40

save_dir = "/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/NPZ_files/"
Fig_dir = '/projekt4/hochatm/smehrdad/GW_localized_uaicon_Simulation/Perturbed_run/Test/Figs/'



EP_V_C = ma.zeros((6, 5437, 40, 120))
EP_M_C = ma.zeros((6, 5437, 40, 120))
EPD_C = ma.zeros((6, 5437, 40, 120))
U_C = ma.zeros((6, 5437, 40, 120))
EP_V_EA = ma.zeros((6, 5437, 40, 120))
EP_M_EA = ma.zeros((6, 5437, 40, 120))
EPD_EA = ma.zeros((6, 5437, 40, 120))
U_EA = ma.zeros((6, 5437, 40, 120))
EP_V_NA = ma.zeros((6, 5437, 40, 120))
EP_M_NA = ma.zeros((6, 5437, 40, 120)) 
EPD_NA = ma.zeros((6, 5437, 40, 120))
U_NA = ma.zeros((6, 5437, 40, 120))
EP_V_HI = ma.zeros((6, 5437, 40, 120)) 
EP_M_HI = ma.zeros((6, 5437, 40, 120))
EPD_HI = ma.zeros((6, 5437, 40, 120))
U_HI = ma.zeros((6, 5437, 40, 120))
Month = ma.zeros((6, 5437))
nn_NA = ma.zeros((6))

os.chdir(save_dir)
loaded = np.load('Zonal_mean_U_EP_Flux_Ens_00.npz')
lat = loaded['lat']
lon = loaded['lon']
plev = loaded['plev']
plev_n = np.where(plev==32500)[0][0]
plev = plev[plev_n:]

for ensembel in range(6):
    if ensembel==4:
        loaded = np.load('Zonal_mean_U_EP_Flux_Ens_0'+str(ensembel)+'.npz')
        Month_t = loaded['Month']
        cut_id = ma.masked_array(loaded['EP_V_NA'][:,plev_n:,:], mask=loaded['EP_V_NA_M'][:,plev_n:,:]).shape[0]
        Month_t_NA = Month_t[:cut_id]
        w_id = seasonal_Winter_id(Month_t)
        w_id_NA = seasonal_Winter_id(Month_t_NA)
        Month[ensembel] = Month_t[w_id]
        EP_V_C[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_C'][w_id,plev_n:,:], mask=loaded['EP_V_C_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_C[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_C'][w_id,plev_n:,:], mask=loaded['EP_M_C_M'][w_id,plev_n:,:]),limit_fz)
        EPD_C[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_C'][w_id,plev_n:,:], mask=loaded['EPD_C_M'][w_id,plev_n:,:]),ep_lim)
        U_C[ensembel,:,:,:] = ma.masked_array(loaded['U_C'][w_id,plev_n:,:], mask=loaded['U_C_M'][w_id,plev_n:,:])
        EP_V_EA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_EA'][w_id,plev_n:,:], mask=loaded['EP_V_EA_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_EA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_EA'][w_id,plev_n:,:], mask=loaded['EP_M_EA_M'][w_id,plev_n:,:]),limit_fz)
        EPD_EA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_EA'][w_id,plev_n:,:], mask=loaded['EPD_EA_M'][w_id,plev_n:,:]),ep_lim)
        U_EA[ensembel,:,:,:] = ma.masked_array(loaded['U_EA'][w_id,plev_n:,:], mask=loaded['U_EA_M'][w_id,plev_n:,:])
        EP_V_NA[ensembel,:w_id_NA.shape[0],:,:] = filtering(ma.masked_array(loaded['EP_V_NA'][w_id_NA,plev_n:,:], mask=loaded['EP_V_NA_M'][w_id_NA,plev_n:,:]),limit_fz)
        EP_M_NA[ensembel,:w_id_NA.shape[0],:,:] = filtering(ma.masked_array(loaded['EP_M_NA'][w_id_NA,plev_n:,:], mask=loaded['EP_M_NA_M'][w_id_NA,plev_n:,:]),limit_fz)
        EPD_NA[ensembel,:w_id_NA.shape[0],:,:] = filtering(ma.masked_array(loaded['EPD_NA'][w_id_NA,plev_n:,:], mask=loaded['EPD_NA_M'][w_id_NA,plev_n:,:]),ep_lim)
        U_NA[ensembel,:w_id_NA.shape[0],:,:] = ma.masked_array(loaded['U_NA'][w_id_NA,plev_n:,:], mask=loaded['U_NA_M'][w_id_NA,plev_n:,:])
        EP_V_HI[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_HI'][w_id,plev_n:,:], mask=loaded['EP_V_HI_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_HI[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_HI'][w_id,plev_n:,:], mask=loaded['EP_M_HI_M'][w_id,plev_n:,:]),limit_fz)
        EPD_HI[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_HI'][w_id,plev_n:,:], mask=loaded['EPD_HI_M'][w_id,plev_n:,:]),ep_lim)
        U_HI[ensembel,:,:,:] = ma.masked_array(loaded['U_HI'][w_id,plev_n:,:], mask=loaded['U_HI_M'][w_id,plev_n:,:])
        nn_NA[ensembel] = w_id_NA.shape[0]
    else:
        loaded = np.load('Zonal_mean_U_EP_Flux_Ens_0'+str(ensembel)+'.npz')
        Month_t = loaded['Month']
        w_id = seasonal_Winter_id(Month_t)
        Month[ensembel] = Month_t[w_id]
        EP_V_C[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_C'][w_id,plev_n:,:], mask=loaded['EP_V_C_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_C[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_C'][w_id,plev_n:,:], mask=loaded['EP_M_C_M'][w_id,plev_n:,:]),limit_fz)
        EPD_C[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_C'][w_id,plev_n:,:], mask=loaded['EPD_C_M'][w_id,plev_n:,:]),ep_lim)
        U_C[ensembel,:,:,:] = ma.masked_array(loaded['U_C'][w_id,plev_n:,:], mask=loaded['U_C_M'][w_id,plev_n:,:])
        EP_V_EA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_EA'][w_id,plev_n:,:], mask=loaded['EP_V_EA_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_EA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_EA'][w_id,plev_n:,:], mask=loaded['EP_M_EA_M'][w_id,plev_n:,:]),limit_fz)
        EPD_EA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_EA'][w_id,plev_n:,:], mask=loaded['EPD_EA_M'][w_id,plev_n:,:]),ep_lim)
        U_EA[ensembel,:,:,:] = ma.masked_array(loaded['U_EA'][w_id,plev_n:,:], mask=loaded['U_EA_M'][w_id,plev_n:,:])
        EP_V_NA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_NA'][w_id,plev_n:,:], mask=loaded['EP_V_NA_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_NA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_NA'][w_id,plev_n:,:], mask=loaded['EP_M_NA_M'][w_id,plev_n:,:]),limit_fz)
        EPD_NA[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_NA'][w_id,plev_n:,:], mask=loaded['EPD_NA_M'][w_id,plev_n:,:]),ep_lim)
        U_NA[ensembel,:,:,:] = ma.masked_array(loaded['U_NA'][w_id,plev_n:,:], mask=loaded['U_NA_M'][w_id,plev_n:,:])
        EP_V_HI[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_V_HI'][w_id,plev_n:,:], mask=loaded['EP_V_HI_M'][w_id,plev_n:,:]),limit_fz)
        EP_M_HI[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EP_M_HI'][w_id,plev_n:,:], mask=loaded['EP_M_HI_M'][w_id,plev_n:,:]),limit_fz)
        EPD_HI[ensembel,:,:,:] = filtering(ma.masked_array(loaded['EPD_HI'][w_id,plev_n:,:], mask=loaded['EPD_HI_M'][w_id,plev_n:,:]),ep_lim)
        U_HI[ensembel,:,:,:] = ma.masked_array(loaded['U_HI'][w_id,plev_n:,:], mask=loaded['U_HI_M'][w_id,plev_n:,:])
        nn_NA[ensembel] = w_id.shape[0]

        
def save_to_hdf5(save_path, lat, lon, plev, EP_V_C, EP_M_C, EPD_C, U_C, EP_V_EA, EP_M_EA, EPD_EA, U_EA, EP_V_NA, EP_M_NA,  EPD_NA, U_NA, EP_V_HI, EP_M_HI, EPD_HI, U_HI, Month, nn_NA ):
    with h5py.File(save_path, 'w') as hdf:
        hdf.create_dataset('lat', data=lat)
        hdf.create_dataset('lon', data=lon)
        hdf.create_dataset('plev', data=plev)
        hdf.create_dataset('EP_V_C', data=EP_V_C)
        hdf.create_dataset('EP_M_C', data=EP_M_C)
        hdf.create_dataset('EPD_C', data=EPD_C)
        hdf.create_dataset('U_C', data=U_C)
        hdf.create_dataset('EP_V_EA', data=EP_V_EA)
        hdf.create_dataset('EP_M_EA', data=EP_M_EA)
        hdf.create_dataset('EPD_EA', data=EPD_EA)
        hdf.create_dataset('U_EA', data=U_EA)
        hdf.create_dataset('EP_V_NA', data=EP_V_NA) 
        hdf.create_dataset('EP_M_NA', data=EP_M_NA) 
        hdf.create_dataset('EPD_NA', data=EPD_NA)
        hdf.create_dataset('U_NA', data=U_NA)
        hdf.create_dataset('EP_V_HI', data=EP_V_HI)
        hdf.create_dataset('EP_M_HI', data=EP_M_HI)
        hdf.create_dataset('EPD_HI', data=EPD_HI)
        hdf.create_dataset('U_HI', data=U_HI)
        hdf.create_dataset('Month', data=Month)
        hdf.create_dataset('nn_NA', data=nn_NA)
    print(f"Data saved to {save_path}")


save_path = os.path.join(save_dir, f"EP_flux_all_ens_data.h5")  # Change extension to .h5 for HDF5 format
save_to_hdf5(save_path, lat, lon, plev, EP_V_C, EP_M_C, EPD_C, U_C, EP_V_EA, EP_M_EA, EPD_EA, U_EA, EP_V_NA, EP_M_NA,  EPD_NA, U_NA, EP_V_HI, EP_M_HI, EPD_HI, U_HI, Month, nn_NA )



# Testing the data
"""
EPD_C[:,:,:,0] = 0
EPD_C[:,:,:,-1] = 0

plev = plev[plev_n:]
Plev_lg = np.log(plev)
Plev_lg1 = Plev_lg[0] - Plev_lg
y_positions = [Plev_lg1[1],Plev_lg1[5],Plev_lg1[9],Plev_lg1[11],Plev_lg1[15],Plev_lg1[20],Plev_lg1[24],Plev_lg1[28],Plev_lg1[32],Plev_lg1[35],Plev_lg1[37],Plev_lg1[39]]
y_positions_a = np.array(y_positions)
y_positions_a = Plev_lg[0] - y_positions_a
y_positions_a = np.exp(y_positions_a)
y_labels = [np.rint(y_positions_a[0]*10000)/1000000,np.rint(y_positions_a[1]*10000)/1000000,np.rint(y_positions_a[2]*10000)/1000000,np.rint(y_positions_a[3]*10000)/1000000,np.rint(y_positions_a[4]*10000)/1000000,np.rint(y_positions_a[5]*10000)/1000000,np.rint(y_positions_a[6]*10000)/1000000,np.rint(y_positions_a[7]*10000)/1000000,np.rint(y_positions_a[8]*10000)/1000000,np.rint(y_positions_a[9]*10000)/1000000,np.rint(y_positions_a[10]*10000)/1000000,np.rint(y_positions_a[11]*10000)/1000000]




plot_climatology( EPD_C[0,1200,:,:], U_C[0,1200,:,:], EP_M_C[0,1200,:,:], EP_V_C[0,1200,:,:], 'Zonal_mean_EP_flux_Climatology')
"""


