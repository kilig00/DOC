import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


# 找不到包的话，需要把包的路径添加到python要搜索的路径中推荐.pth文件

def cal_performance(x, y):
    """x and y are numpy arrays"""
    if np.ndim(x) != np.ndim(y):
        raise ValueError("x and y must have the same dimensionality")
    if np.ndim(x) == 2:
        x = x.squeeze()  # convert to 1D array
        y = y.squeeze()

    R = np.corrcoef(x, y)  # correlation matrix, x and y are 1D array. 如果是2D array，会计算每一列的相关系数，矩阵大了会卡死
    R2 = R[0, 1] ** 2
    rmse = np.sqrt(np.mean((x - y) ** 2))  # units: same as x and y
    mape = np.mean(abs((x - y) / x)) * 100  # units: %
    n = x.size
    return R2, rmse, mape, n


def cal_performance_prc(x, y):
    """
    x and y are numpy arrays
    Returns:
        R2   : coefficient of determination (Pearson r^2)
        rmse : root mean square error
        mape : mean absolute percentage error (%)
        n    : sample size
        p    : p-value of Pearson correlation
    """
    if np.ndim(x) != np.ndim(y):
        raise ValueError("x and y must have the same dimensionality")

    if np.ndim(x) == 2:
        x = x.squeeze()  # convert to 1D array
        y = y.squeeze()

    # remove nan / inf (这里做一次兜底，防止外面漏掉)
    flag = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y))
    x = x[flag]
    y = y[flag]

    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan, n, np.nan

    # Pearson correlation
    _, p = pearsonr(x, y)

    R = np.corrcoef(x, y)  # correlation matrix, x and y are 1D array. 如果是2D array，会计算每一列的相关系数，矩阵大了会卡死
    R2 = R[0, 1] ** 2

    rmse = np.sqrt(np.mean((x - y) ** 2))      # same unit as x, y
    mape = np.mean(np.abs((x - y) / x)) * 100  # %

    return R2, rmse, mape, n, p


def plot_scatter(ax, y_true, y_pred, label, point_size=100, is_density=False, colorbar_norm=None):
    font_dic = {'family': 'arial',
                'weight': 'normal',
                'size': 28}

    bbox_dic = {'ec': "grey",
                'lw': 0,
                'facecolor': 'white',
                'alpha': 0}

    y_true = np.array(y_true).reshape(1, -1)
    y_pred = np.array(y_pred).reshape(1, -1)
    # delete nan
    flag_nan = np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred)
    y_true = y_true[~flag_nan].reshape(1, -1)
    y_pred = y_pred[~flag_nan].reshape(1, -1)

    # calculate performance metrics
    def p_threshold_text(p):
        if p < 0.01:
            return "≤ 0.01"
        elif p < 0.05:
            return "≤ 0.05"
        else:
            return "> 0.05"
    
    R2, rmse, mape, n, p = cal_performance_prc(y_true, y_pred)

    metrics = [f"$R^{2}$ = {R2:.2f}",
               f"RMSE = {rmse:.2f}",
               f"MAPE = {mape:.1f} %",
               f"N = {n}"]

    max = np.max([np.max(y_true), np.max(y_pred)]) + 0.7

    # plot scatter
    if is_density == False:
        if label == 'training':
            ax.scatter(y_true, y_pred, label=label,
                    facecolors='#95CDDC', edgecolors='#5081BA', linewidths=1,
                    s=100, marker='o')
            ax.text(0.04, 0.95, "\n".join(metrics), transform=ax.transAxes,
                    ha='left', va='top',
                    fontdict=font_dic, color='#5081BA', bbox=bbox_dic)
                    # --- p-value（同一行，向右偏移，字号更小，斜体 p） ---
            ax.text(0.305, 0.945,   
                rf"($\mathit{{p}}$ {p_threshold_text(p)})",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=22,
                family='arial',
                color='#5081BA', bbox=bbox_dic)
            
        else:
            ax.scatter(y_true, y_pred, label=label,
                    facecolors='#F9C095', edgecolors='#E26C25', linewidths=1,
                    s=100, marker='s', alpha=0.6)
            ax.text(0.53, 0.28, "\n".join(metrics), transform=ax.transAxes,
                    ha='left', va='top',
                    fontdict=font_dic, color='#E26C25', bbox=bbox_dic)
            ax.text(0.795, 0.275,   
                rf"($\mathit{{p}}$ {p_threshold_text(p)})",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=22,
                family='arial',
                color='#E26C25', bbox=bbox_dic)
        ax.plot([0, max], [0, max], zorder=0, color='grey', linewidth=1, linestyle='--')

    if is_density == True:
        ax.hist2d(y_true.squeeze(), y_pred.squeeze(), bins=100, cmin=1, cmap=plt.get_cmap('jet'))
        # sns.set_style("white")
        # sns.kdeplot(x=y_true.squeeze(), y=y_pred.squeeze(),
        #             cmap="RdBu_r",
        #             fill=True,
        #             thresh=0.1,
        #             # thresh：取值范围通常在0到1之间，这个值表示密度估计的最小比例阈值。例如，thresh=0.1 意味着只有那些密度值大于或等于密度最大值的10%的区域才会被包含在绘制的等高线中。
        #             bw_adjust=0.2,
        #             ax=ax,
        #             norm=colorbar_norm)

    ax.set_xlim(0, max)
    ax.set_ylim(0, max)
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_tick_params(width=1.5, length=5, direction='out')
    ax.yaxis.set_tick_params(width=1.5, length=5, direction='out')
    ax.tick_params(axis='both', which='minor', length=5, color='black', labelsize=0)

    

    for axis in ['left', 'right', 'bottom', 'top']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('k')

    ax.set_xlabel('in-situ data of DOC [mg/L])', fontdict=font_dic)
    ax.set_ylabel('Predicted DOC [mg/L]', fontdict=font_dic)
    ax.legend(prop=font_dic, frameon=True, loc='upper right')
    ax.set_aspect('equal')


def crop_images(lon, lat, spectrum_3d, flag, coordinates):
    lonmin, lonmax, latmin, latmax = coordinates
    id = (lon > lonmin) & (lon < lonmax) & (lat > latmin) & (lat < latmax)  # XMB
    idx_col = np.where(np.logical_not(np.all(id == False, 0)))[0]
    idx_row = np.where(np.logical_not(np.all(id == False, 1)))[0]

    top_row = idx_row[0]
    bottom_row = idx_row[-1]
    left_col = idx_col[0]
    right_col = idx_col[-1]

    lon_crop = lon[top_row:bottom_row, left_col:right_col]
    lat_crop = lat[top_row:bottom_row, left_col:right_col]

    spectrum_3d_crop = []
    for idx_band in range(spectrum_3d.shape[2]):
        spectrum_3d_crop.append(spectrum_3d[top_row:bottom_row, left_col:right_col, idx_band])

    spectrum_3d_crop = np.stack(spectrum_3d_crop, axis=2)

    if flag:
        flag_crop = flag[top_row:bottom_row, left_col:right_col]
        return lon_crop, lat_crop, spectrum_3d_crop, flag_crop
    else:
        return lon_crop, lat_crop, spectrum_3d_crop


def get_satellite_band(satellite_type, bands_type='all'):
    """
    satellite_type:
    - sentinel-3 a
    - sentinel-3 b
    - sentinel-2 a
    - sentinel-2 b
    - landsat-7
    - landsat-8
    - landsat-9
    """

    bands = None
    # Acolite band name
    if bands_type == 's2s3 common band':
        # bands identical to S3
        if (satellite_type.lower() == 'sentinel-3 a') or (satellite_type.lower() == 'sentinel-3 b'):
            bands = [443, 490, 560, 620, 709, 754, 779, 865]  # S3A/B, the common bands with S2 bands are same in S3A/B
        elif (satellite_type.lower() == 'sentinel-2 a'):
            bands = [443, 492, 560, 665, 704, 740, 783, 865]  # S2A
        elif (satellite_type.lower() == 'sentinel-2 b'):
            bands = [442, 492, 559, 665, 704, 739, 780, 864]  # S2B
        else:
            raise ValueError("Satellite type not found")

    elif bands_type == 'all':
        if (satellite_type.lower() == 'sentinel-3 a'):
            bands = [400, 412, 443, 490, 510, 560, 620, 665, 674, 682, 709, 754, 768, 779, 865, 884, 1016]  # S3A
        elif (satellite_type.lower() == 'sentinel-3 b'):
            bands = [401, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754, 768, 779, 865, 884, 1016]  # S3B
        elif (satellite_type.lower() == 'sentinel-2 a'):
            bands = [443, 492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]  # S2A
        elif (satellite_type.lower() == 'sentinel-2 b'):
            bands = [442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186]  # s2B
        elif (satellite_type.lower() == 'landsat-7'):
            bands = [479, 561, 661, 720, 835, 1650, 2208]
        elif (satellite_type.lower() == 'landsat-8'):
            bands = [443, 483, 561, 592, 655, 865, 1609, 2201]  # L8
        elif (satellite_type.lower() == 'landsat-9'):
            bands = [443, 482, 561, 594, 654, 865, 1608, 2201]  # L9
        else:
            raise ValueError("Satellite type not found")

    return bands


def get_satellite_name_acolite(fn):
    if 'S3A' in fn:
        return 'sentinel-3 a'
    elif 'S3B' in fn:
        return 'sentinel-3 b'
    elif 'S2A' in fn:
        return 'sentinel-2 a'
    elif 'S2B' in fn:
        return 'sentinel-2 b'
    elif 'L8' in fn:
        return 'landsat-8'
    elif 'L9' in fn:
        return 'landsat-9'
    else:
        raise ValueError("Satellite type not found")


def read_sentinel_L2Acolite(fullpath, satellite_type, bands_type='all', cloud_mask=True, is_crop=False,
                            coordinates=None, is_l2_flags=False):
    """
    Read Sentinel-3 L2 Acolite data
    :satellite: sentinel-2 or sentinel-3
    :return: lon lat rhorc(row, col, n_band)
    """

    ncdata = nc.Dataset(fullpath)
    lon = ncdata.variables['lon'][:]
    lat = ncdata.variables['lat'][:]

    if is_l2_flags:  # l2flags每幅图的结果不一样，需要手动查看
        l2_flags = ncdata.variables['l2_flags'][:]
        # l2_flags is different for s2 and s3
        if 'sentinel-3' in satellite_type.lower():
            l2_flags_logic = np.where(l2_flags == 0, 1,
                                      0)  # 不等于0，就是质量不好的像元. mask flag > 0, remove the bad quality pixels
        elif 'sentinel-2' in satellite_type.lower():
            l2_flags_logic = np.where(l2_flags == 1, 1, 0)  # 查看flag后结果，水体是1。但还是会有一部分s2的陆地之没有去除，需要手动去除

    # cloud mask [如果没有rhorc_865就不进行云掩膜]
    if ('rhorc_865' in ncdata.variables.keys()) or ('rhorc_864' in ncdata.variables.keys()) and cloud_mask:
        try:
            flag_cloud = ncdata.variables['rhorc_865'][:]
        except:
            flag_cloud = ncdata.variables['rhorc_864'][:]
        flags_cloud_logic = np.where(flag_cloud > 0.2, 1, 0)
        # rhorc_865 > 0.2 是有云 flag_cloud_logic 1 代表云
        # s2和s3用一样的阈值
    else:
        cloud_mask = False

    # Rrs_np = []
    rhorc_np = []
    bands = get_satellite_band(satellite_type, bands_type)

    for band in bands:
        # Rrs_band = ncdata.variables['Rrs_' + str(band)][:]
        rhorc_band = ncdata.variables['rhorc_' + str(band)][:]
        # if is_l2_flags: # l2flags每幅图的结果不一样，需要手动查看
        #     rhorc_band = np.where(l2_flags_logic, rhorc_band, np.nan)  # l2_flags_logic 1 is good quality pixel
        if cloud_mask:
            rhorc_band = np.where(flags_cloud_logic, np.nan, rhorc_band)  # flag_cloud_logic 1 is cloud

        # Rrs_np.append(Rrs_band)
        rhorc_np.append(rhorc_band)

    # Rrs_np_3d = np.stack(Rrs_np, axis=2)
    rhorc_np_3d = np.stack(rhorc_np, axis=2)

    if is_crop and coordinates is not None:
        # coordinates = [117.95, 118.54, 24.33, 24.68] # XMB
        if not is_l2_flags:
            lon_crop, lat_crop, rhorc_np_crop = crop_images(lon, lat, rhorc_np_3d, flag=None, coordinates=coordinates)
            lon, lat, rhorc_np_3d = lon_crop, lat_crop, rhorc_np_crop  # 重新赋值
        else:
            lon_crop, lat_crop, rhorc_np_crop, l2_flags_crop = crop_images(lon, lat, rhorc_np_3d, l2_flags_logic,
                                                                           coordinates=coordinates)
            lon, lat, rhorc_np_3d, l2_flags = lon_crop, lat_crop, rhorc_np_crop, l2_flags_crop

    if is_l2_flags:
        return lon, lat, rhorc_np_3d, l2_flags
    else:
        return lon, lat, rhorc_np_3d


def extract_spectral(lon, lat, lon_sat, lat_sat, spectral_sat, how='three_by_three', judge_edge_thre=0.01,
                     is_process=True):
    # match points
    dist = np.sqrt((lon_sat - lon) ** 2 + (lat_sat - lat) ** 2)

    # if ob_data point outside the figure pass it, 0.01 is 1,113 m (s2 10m resolution s3 300m resolution)
    if np.min(dist) > judge_edge_thre:
        return (None, None, None)

    irows, icols = np.where(dist == np.min(dist))  # choose one of the min dist
    irow, icol = irows[0], icols[0]  # 如果有多个最近的，选第一个

    if how == 'one_by_one':

        return [spectral_sat[irow, icol, :]], [lon_sat[irow, icol]], [lat_sat[irow, icol]]

    elif how == 'three_by_three':
        try:  # 如果不在边缘的话
            match_spectrum = np.full((9, spectral_sat.shape[2]), np.nan)
            match_lons = np.full((9, 1), np.nan)
            match_lats = np.full((9, 1), np.nan)
            idx_spec = 0
            for irow_temp in [irow - 1, irow, irow + 1]:
                for icol_temp in [icol - 1, icol, icol + 1]:
                    match_spectrum[idx_spec, :] = spectral_sat[irow_temp, icol_temp, :]
                    match_lons[idx_spec] = lon_sat[irow_temp, icol_temp]
                    match_lats[idx_spec] = lat_sat[irow_temp, icol_temp]

                    idx_spec += 1
            if is_process:
                return (np.nanmean(match_spectrum, axis=0, keepdims=True),
                        np.nanmean(match_lons, keepdims=True),
                        np.nanmean(match_lats, keepdims=True))
            else:
                return match_spectrum, match_lons, match_lats
        except:
            return [spectral_sat[irow, icol, :]], [lon_sat[irow, icol]], [lat_sat[irow, icol]]


def match_insitu_satspectrum(df_sample, lon_sat, lat_sat, spectral_sat, how='three_by_three',
                             affix_name='Rrs', judge_edge_thre=0.01, is_process=True, list_bands=None):
    """
    df_sample: station data (must contain Id, lon, lat)
    list_bands: list of bands to be extracted

    """

    def get_coordinate_columns(df):
        """get the coordinate columns in the DataFrame"""
        lon_options = ['经度', 'lon', 'longitude']  # lon column name
        lat_options = ['纬度', 'lat', 'latitude']  # lat column name
        lon_column = next((col for col in lon_options if col in df.columns), None)
        lat_column = next((col for col in lat_options if col in df.columns), None)
        if not lon_column or not lat_column:
            raise ValueError("DataFrame必须包含经度和纬度列")
        return lon_column, lat_column

    lon_column, lat_column = get_coordinate_columns(df_sample)
    rs_list = []
    for index, row in df_sample.iterrows():
        # lon, lat = df_sample.loc[index, ['经度', '纬度']]
        lon, lat = row[lon_column], row[lat_column]

        array_spectrum, array_lon, array_lat = extract_spectral(lon, lat, lon_sat, lat_sat, spectral_sat, how=how,
                                                                judge_edge_thre=judge_edge_thre, is_process=is_process)
        if array_spectrum is None:
            continue

        df_spectrum = pd.DataFrame(array_spectrum)
        df_lon = pd.DataFrame(array_lon)
        df_lat = pd.DataFrame(array_lat)

        df_match = pd.concat([df_spectrum, df_lon, df_lat], axis=1)

        df_match.columns = ([affix_name + '_' + str(band) for band in list_bands] +
                            ['lon_' + affix_name, 'lat_' + affix_name])
        df_match['Id'] = row['Id']
        rs_list.append(df_match)

    return pd.concat(rs_list, axis=0) if rs_list else None


def save_nc_file(save_name, lon, lat, DOC_arr_3d):
    f_w = nc.Dataset(save_name, 'w', format='NETCDF4')

    # define dimensions
    longs = f_w.createDimension('longitude', size=lon.shape[1])
    lats = f_w.createDimension('latitude', size=lon.shape[0])

    # create variables
    lat_w = f_w.createVariable('lat', np.float32, ('latitude', 'longitude'))
    lon_w = f_w.createVariable('lon', np.float32, ('latitude', 'longitude'))
    DOC_w = f_w.createVariable('DIP', np.float32, ('latitude', 'longitude'))

    lon_w[:] = lon
    lat_w[:] = lat
    DOC_w[:] = DOC_arr_3d
    f_w.close()


def save_resampled_nc_file(save_name, lon, lat, spectrum_3d, bands_name):
    f_w = nc.Dataset(save_name, 'w', format='NETCDF4')

    # Define dimensions
    longs = f_w.createDimension('longitude', size=lon.shape[1])
    lats = f_w.createDimension('latitude', size=lon.shape[0])

    # Create variables for latitude and longitude
    lat_w = f_w.createVariable('lat', np.float32, ('latitude', 'longitude'))
    lon_w = f_w.createVariable('lon', np.float32, ('latitude', 'longitude'))

    # Initialize dictionary to hold variable references
    band_vars = {}

    # Create variables for each band
    for band in bands_name:
        band_vars[band] = f_w.createVariable(f'rhorc_{band}', np.float32, ('latitude', 'longitude'))

    # Assign values to latitude and longitude variables
    lon_w[:] = lon
    lat_w[:] = lat

    # Assign values to each spectral band variable
    for i, band in enumerate(bands_name):
        band_vars[band][:] = spectrum_3d[:, :, i]

    # Close the file to write changes to disk
    f_w.close()


