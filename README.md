
# DOC

---

## 论文 | Publication

本代码对应论文：  
This code accompanies the following paper:
本代码对应已投稿论文（尚未正式发表）： 
This repository contains code and data processing scripts for the submitted manuscript “High-resolution satellite retrieval of dissolved organic carbon in coastal waters using machine learning: Application to Dongshan Bay”. Lingling Li, Xiaolong Yu, Wendian Lai, Nengwang Chen, Weidong Guo, Hongyan Bao, Guizhi Wang, Shuiying Huang, Wenlu Lan, Xiaoyan Peng, Zhongping Lee.

---

## 概述 | Overview

- **脚本 01–03**：基于 Acolite 在 Python 中调用 Google Earth Engine (GEE) API，实现实测点与卫星数据的匹配及后续数据处理。  
  **Scripts 01–03**: Match in-situ points with satellite data and perform subsequent data processing using Acolite to call the Google Earth Engine (GEE) API in Python.

- **train-model-DOC-all-20241009.py**：基于 AutoGluon 自动机器学习框架进行模型训练。  
  **train-model-DOC-all-20241009.py**: Model training based on the AutoGluon automated machine learning framework.

---

## 文件说明 | File Description

| 文件 File | 说明 Description |
|-----------|------------------|
| `01Acolite+GEE+csv_forC_20250406.py` | Acolite + GEE：生成/处理与实测点匹配的 CSV。 Acolite + GEE: generate/process CSV matched with in-situ points. |
| `02mergeCSV.py` | 合并 CSV 数据。 Merge CSV files. |
| `03modified_bandName.py` | 修改波段名称等后处理。 Post-processing such as modifying band names. |
| `train-model-DOC-all-20241009.py` | 基于 AutoGluon 的 DOC 模型训练。 DOC model training with AutoGluon. |
| `my_utils.py` / `utils.py` | 项目用工具函数。 Project utility functions. |

---

## 依赖与运行 | Dependencies & Usage

- Python 3.9
- Acolite 20220222.0
- Google Earth Engine API 1.4.0
- Autogluon 1.1.1
