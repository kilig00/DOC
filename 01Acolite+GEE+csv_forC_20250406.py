import ee
import os
import pandas as pd
import acolite as ac
import tempfile
import shutil
import time
import logging
import logging.handlers
from multiprocessing import Pool
from osgeo import gdal

    
# import multiprocessing
 
# !pip install earthengine-api
# !earthengine authenticate --quiet
# conda install -c conda-forge numpy matplotlib scipy gdal pyproj scikit-image pyhdf pyresample netcdf4 h5py requests pygrib astropy cartopy
# 安装上面的库，numpy必须是小于2.0版本

# # 初始化 Earth Engine
# ee.Initialize()



try:
    ee.Authenticate() # ee.Authenticate(force=True)
    ee.Initialize(project='xx')
    print("Earth Engine 初始化成功！")
except Exception as e:
    print("初始化 Earth Engine 时出错:", e)
    print("请尝试先运行 ee.Authenticate() 来完成身份验证。")

def ee_worker_init():
    try:
        ee.Initialize()
        print("子进程Earth Engine 初始化成功！")
    except Exception as e:
        print("子进程初始化 Earth Engine 时出错:", e)


def process_point(index, point, sources, sdate, edate, output_csv):

    temp_dir = tempfile.mkdtemp()

    # 创建临时目录下的输入、输出路径
    temp_input = os.path.join(temp_dir, "input_nc")
    temp_output_nc = os.path.join(temp_dir, "output_acolite")
    temp_output_csv = os.path.join(temp_dir, "output_csv")

    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output_nc, exist_ok=True)
    os.makedirs(temp_output_csv, exist_ok=True)

    # 将当前点的数据写入临时文件
    point_list_dir = os.path.join(temp_dir, "temp_point_list.csv")
    point_df = pd.DataFrame([point])
    point_df.to_csv(point_list_dir, encoding='utf-8-sig', index=False)
    
    try:
        # 调用 acolite_gee_run 函数
        ac.acolite_gee_run.acolite_gee_run(index, point, temp_input, temp_output_nc, temp_output_csv, sources, sdate, edate)
        # 处理完的文件（例如 CSV）可以复制到主输出目录
        for file in os.listdir(temp_output_csv):
            full_file_path = os.path.join(temp_output_csv, file)
            if os.path.isfile(full_file_path):
                shutil.copy(full_file_path, output_csv)
    finally:
        # 清理临时工作目录
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    input = "E:/LLL/01C/Data/xx/input_nc"
    output_nc = "E:/LLL/01C/Data/xx/output_acolite"
    output_csv = "E:/LLL/01C/Data/xx/output_csv"
    sources = ['Sentinel-2']
    point_list_dir = "E:/LLL/01C/Data/xx/test.csv"
    sdate = "2022-07-01" # "2021-03-12"
    edate = "2023-03-31"
    # ac.acolite_gee_run.acolite_gee_run(input, output_nc, output_csv, sources, point_list_dir, sdate, edate)

    # 读取点数据    
    point_list = pd.read_csv(point_list_dir,encoding='utf-8')

    # 获取 output_csv 目录中的文件名（假设文件名为 index）
    existing_files = set(os.listdir(output_csv))

    # 创建任务列表
    # tasks = [(index, point, sources, sdate, edate, output_csv) for index, point in point_list.iterrows()]
    tasks = []
    for index, point in point_list.iterrows():
        # 检查当前 index 是否在 existing_files中
        if f"{index}.csv" not in existing_files:
            tasks.append((index, point, sources, sdate, edate, output_csv))
            print(f'缺失编号是 :{index}')

    # 如果 tasks 不为空则执行多进程处理任务，否则打印消息
    if tasks:
        with Pool(processes=28) as pool:
            pool.starmap(process_point, tasks)
            pool.close()
            pool.join()
    else:
        print("没有未执行的数据，结束！")