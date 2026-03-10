import pandas as pd

def replace_band_names(input_csv_path, output_csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
    
    # 定义 S2A 和 S2B 波段到 B1, B2, ... 的映射
    S2A_MSI_bandSets = {
        'rhorc_443': 'B1',
        'rhorc_492': 'B2',
        'rhorc_560': 'B3',
        'rhorc_665': 'B4',
        'rhorc_704': 'B5',
        'rhorc_740': 'B6',
        'rhorc_783': 'B7',
        'rhorc_833': 'B8',
        'rhorc_865': 'B8A',
        'rhorc_1614': 'B11',
        'rhorc_2202': 'B12'
    }
    
    S2B_MSI_bandSets = {
        'rhorc_442': 'B1',
        'rhorc_492': 'B2',
        'rhorc_559': 'B3',
        'rhorc_665': 'B4',
        'rhorc_704': 'B5',
        'rhorc_739': 'B6',
        'rhorc_780': 'B7',
        'rhorc_833': 'B8',
        'rhorc_864': 'B8A',
        'rhorc_1610': 'B11',
        'rhorc_2186': 'B12'
    }
    
    # 合并所有卫星波段的映射
    bandSets = {**S2A_MSI_bandSets, **S2B_MSI_bandSets}
    
    # 重命名列
    df.rename(columns=bandSets, inplace=True)

    # 创建一个新 DataFrame 来存储最终的列
    final_df = pd.DataFrame()

    # 处理所有重复的列，通过取非空值合并并只保留一列
    for band in set(bandSets.values()):
        # 获取所有具有当前波段名称的列
        band_columns = df.filter(regex=f'^{band}$').columns
        if len(band_columns) > 1:
            # 合并列，取第一个非空值
            final_df[band] = df[band_columns].bfill(axis=1).iloc[:, 0]
        elif len(band_columns) == 1:
            final_df[band] = df[band_columns[0]]
    
    # 保留所有其他不在 bandSets 中的列
    remaining_columns = df.columns.difference(final_df.columns)
    final_df = pd.concat([final_df, df[remaining_columns]], axis=1)

    # # 按指定顺序排列列
    # columns_order = [
    #     "lon", "lat", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", 
    #     "l2_flags", "POI", "img_datetime", "in-situ_datetime", "img_S2",
    #     "CLOUD_COVER", "DOC-mg/L", "Chla-μg/L",  "Salinity", "Temp-℃", 
    #     "time_windows-h", "time_difference-d", "Location", "StationName", "WaterDepth-m", "SampleDepth-m",
    #     "Note1", "Note2", "PIs", "From_what_table"
    # ]


    
    # 确保所有列都存在于final_df中，且列的顺序与columns_order一致
    columns_order = [col for col in columns_order if col in final_df.columns]
    final_df = final_df[columns_order]

    # 去除 POI 和 img_datetime 都相同的重复行
    final_df.drop_duplicates(subset=['POI', 'img_datetime'], keep='first', inplace=True)

    # 按 POI 和 img_datetime 排序
    final_df.sort_values(by=['POI', 'img_datetime'], inplace=True)

    # 保存为新的 CSV 文件
    final_df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)

# 示例用法:
input_csv_path = 'E:/LLL/01C/Data/GEE_S2_DIC/merge/S2_matchUp.csv'
output_csv_path = 'E:/LLL/01C/Data/GEE_S2_DIC/merge/S2_matchUp_reviseName.csv'
replace_band_names(input_csv_path, output_csv_path)
print('已修改波段名')
