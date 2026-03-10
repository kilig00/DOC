import pandas as pd
import glob
import os

# # DOC
# columns = [
#     "lon", "lat", "rhorc_442", "rhorc_443", "rhorc_492", "rhorc_559", "rhorc_560", 
#     "rhorc_665", "rhorc_704", "rhorc_739", "rhorc_740", "rhorc_780", "rhorc_783", 
#     "rhorc_833", "rhorc_864", "rhorc_865", "rhorc_1610", "rhorc_1614", "rhorc_2186", 
#     "rhorc_2202", "l2_flags", "POI", "img_datetime", "in-situ_datetime", "time_windows-h", 'time_difference-d',
#     "CLOUD_COVER", "DOC-mg/L", "Chla-μg/L", "Salinity", "Temp-℃",
#     "Location", "img_S2", "StationName", "WaterDepth-m", "SampleDepth-m", "Note1", "Note2",
#     "PIs", "From_what_table"
# ]



# 找到所有CSV文件的路径
csv_files = glob.glob("E:/LLL/01C/Data/xx/output_csv/*.csv")
# csv_files = glob.glob("F:/LLL/01C/Data/GEE_S2-test/output_csv_Rrs/*.csv")

# 创建一个空的DataFrame，并指定列
merged_df = pd.DataFrame(columns=columns)

# 读取每个CSV文件并追加到merged_df中
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if df.empty:
            continue  # 如果文件是空的，跳过
        # 确保每个DataFrame都有所有的列
        df = df.reindex(columns=columns, fill_value='')
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    except pd.errors.EmptyDataError:
        # 如果文件内容为空或读取失败，跳过
        continue

# 保存合并后的CSV文件
merged_df.to_csv("E:/LLL/01C/Data/xx/merge/S2_matchUp.csv", encoding='utf-8-sig',  index=False)

print("已合并")