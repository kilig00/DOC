import netCDF4 as nc
import numpy as np


def crop_images(lon, lat, l2_flags, spectrum_3d):
    # id = (lon > 117.4) & (lon < 117.58) & (lat > 23.9) & (lat < 23.96)  # DSB
    # id = (lon > 117.38) & (lon < 117.65) & (lat > 23.7) & (lat < 23.98)  # DSB_all
    # id = (lon > 117.38) & (lon < 117.63) & (lat > 23.84) & (lat < 23.98)  # DSB_big
    # id = (lon > 117.26) & (lon < 117.81) & (lat > 23.54) & (lat < 24.1)  # DSB_all_big
    # id = (lon > 117.28) & (lon < 118.00) & (lat > 23.50) & (lat < 24.15)  # DSB_20240821
    # id = (lon > 117.31) & (lon < 117.66) & (lat > 23.67) & (lat < 24.00)  # DSB_20240923
    # id = (lon > 117.989) & (lon < 118.494) & (lat > 24.324) & (lat < 24.671)  # XMB_20260205论文增加内容
    id = (lon > 108.283) & (lon < 108.764) & (lat > 21.597) & (lat < 21.955)  # BBG_20260205论文增加内容
    idx_col = np.where(np.logical_not(np.all(id == False, 0)))[0]
    idx_row = np.where(np.logical_not(np.all(id == False, 1)))[0]

    top_row = idx_row[0]
    bottom_row = idx_row[-1]
    left_col = idx_col[0]
    right_col = idx_col[-1]

    lon_crop = lon[top_row:bottom_row, left_col:right_col]
    lat_crop = lat[top_row:bottom_row, left_col:right_col]
    l2_flags_crop = l2_flags[top_row:bottom_row, left_col:right_col]

    spectrum_3d_crop = []
    for idx_band in range(spectrum_3d.shape[2]):
        spectrum_3d_crop.append(spectrum_3d[top_row:bottom_row, left_col:right_col, idx_band])

    spectrum_3d_crop = np.stack(spectrum_3d_crop, axis=2)

    return lon_crop, lat_crop, l2_flags_crop, spectrum_3d_crop


def read_S2_L2Acolite_crop(S2_L2Acolite_fullpath):
    """
    Read acolite Sentinel-2 data [partial bands, identical to s3]
    :param L2W
    :return: lon lat Rrs(row, col, n_band)
    """

    ncdata = nc.Dataset(S2_L2Acolite_fullpath)
    lon = ncdata.variables['lon'][:]
    lat = ncdata.variables['lat'][:]
    l2_flags = ncdata.variables['l2_flags'][:]

    # Acolite band name
    if "S2A" in S2_L2Acolite_fullpath:
        bands = [443, 492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202] # bands identical to S3
    else:
        bands = [442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186]  # s2B

    # Rrs_np = []
    rhorc_np = []

    for band in bands:
        # Rrs_band = ncdata.variables['Rrs_' + str(band)][:]
        rhorc_band = ncdata.variables['rhorc_' + str(band)][:]

        # Rrs_np.append(Rrs_band)
        rhorc_np.append(rhorc_band)

    # Rrs_np_3d = np.stack(Rrs_np, axis=2)
    rhorc_np_3d = np.stack(rhorc_np, axis=2)
    # return lon, lat, Rrs_np_3d, rhorc_np_3d

    lon_crop, lat_crop, l2_flags_crop, rhorc_np_crop = crop_images(lon, lat, l2_flags, rhorc_np_3d)
    return lon_crop, lat_crop, l2_flags_crop, rhorc_np_crop


def read_S3_L2Acolite_crop(S3_L2Acolite_fullpath):
    """
    Read Sentinel-3 L2 Acolite data
    :param L2W
    :return: lon lat Rrs(row, col, n_band)
    """

    ncdata = nc.Dataset(S3_L2Acolite_fullpath)
    lon = ncdata.variables['lon'][:]
    lat = ncdata.variables['lat'][:]

    # Acolite band name
    if "S3A" in S3_L2Acolite_fullpath:
        bands = [443, 490, 560, 620, 709, 754, 779, 865]  # S3A
    else:
        bands = [443, 490, 560, 620, 709, 754, 779, 865]   # S3B

    # Rrs_np = []
    rhorc_np = []

    for band in bands:
        # Rrs_band = ncdata.variables['Rrs_' + str(band)][:]
        rhorc_band = ncdata.variables['rhorc_' + str(band)][:]

        # Rrs_np.append(Rrs_band)
        rhorc_np.append(rhorc_band)

    # Rrs_np_3d = np.stack(Rrs_np, axis=2)
    rhorc_np_3d = np.stack(rhorc_np, axis=2)
    lon_crop, lat_crop, rhorc_np_crop = crop_images(lon, lat, rhorc_np_3d)

    return lon_crop, lat_crop, rhorc_np_crop


def save_nc_file(save_name, lon, lat, DOC_arr_3d):
    f_w = nc.Dataset(save_name, 'w', format='NETCDF4')

    # define dimensions
    longs = f_w.createDimension('longitude', size=lon.shape[1])
    lats = f_w.createDimension('latitude', size=lon.shape[0])

    # create variables
    lat_w = f_w.createVariable('lat', np.float32, ('latitude', 'longitude'))
    lon_w = f_w.createVariable('lon', np.float32, ('latitude', 'longitude'))
    DOC_w = f_w.createVariable('DOC', np.float32, ('latitude', 'longitude'))

    lon_w[:] = lon
    lat_w[:] = lat
    DOC_w[:] = DOC_arr_3d
    f_w.close()
