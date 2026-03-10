from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
from my_utils import *


def train_autogluon_final(train_data, val_data):
    # 确保 'img_datetime' 列为 datetime 类型
    train_data['img_datetime'] = pd.to_datetime(train_data['img_datetime'])
    val_data['img_datetime'] = pd.to_datetime(val_data['img_datetime'])

    label1 = 'DOC-mg/L'  
    drop_columns = [#'B7', ########################################################################################
                    'lon', 'lat', 'B11', 'B12', 'POI', 'l2_flags', 'in-situ_datetime', 'img_S2',
                    'CLOUD_COVER', 'Chla-μg/L', 'Salinity', 'Temp-℃', 
                    'time_windows-h', 'time_difference-d', 'Location',
                    'StationName', 'WaterDepth-m', 'SampleDepth-m', 'Note1',
                    'Note2', 'PIs', 'From_what_table']      # "Id"等列是不需要放到训练里面的                                                                                                                                                                                                                                    
    # add_drop_labels = ['lon', 'lat']
    log_label = [label1]
    
    # log some features of train data
    for label in log_label:
        train_data[label] = np.log(train_data[label])
        val_data[label] = np.log(val_data[label])
    
    # Combine train_data and val_data for bagged mode
    train_val_data = pd.concat([train_data, val_data], ignore_index=True)

    predictor = TabularPredictor(label=label1, 
                                 path=f'E:/LLL/01C/Program/NN/AutogluonModelDOC-20250407-2.5d' #_rmv_B7####################################################
                                #  eval_metric='root_mean_squared_error' # default:'root_mean_squared_error';'mean_absolute_percentage_error'
                                 ).fit(
                                        train_val_data.drop(columns=drop_columns),
                                        hyperparameters='multimodal',
                                        num_stack_levels=1, num_bag_folds=5,
    )  # num_bag_folds k-则交叉bagging 训练5个模型，输出5个的平均值,一般5应该差不多了

    '''
    # predictor = TabularPredictor.load('E:/LLL/Program/NN/AutogluonModelDOC/')
    # # 查看模型训练摘要
    # summary = predictor.fit_summary()
    # print(summary)
    # # 查看模型详细信息
    # info = predictor.info()
    # print(info)
    # # 查看 trainer 信息
    # trainer_info = predictor._trainer.get_info()
    # print(trainer_info)
    '''


    # Test
    preds_train = predictor.predict(train_data.drop(columns=drop_columns))
    preds_val = predictor.predict(val_data.drop(columns=drop_columns))
    
    # 绘制true和preds的散点图
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    font_dic = {"size": 32,
                "family": "Arial"
                }
    
    # plot
    _, axs = plt.subplots(1, 1, figsize=(10, 8))
    plot_scatter(axs, np.exp(train_data[label1]), np.exp(preds_train), 'training', 0.5)
    plot_scatter(axs, np.exp(val_data[label1]), np.exp(preds_val), 'validation', 0.5)
    # axs.set_title('DOC [mg/L]', fontdict=font_dic)
    axs.set_xlabel(r'$\mathit{in}\,\text{-}\,\mathit{situ}$ DOC (mg/L)', fontdict=font_dic)  # 设置横坐标标签
    axs.set_ylabel('Estimated DOC (mg/L)', fontdict=font_dic)  # 设置纵坐标标签

    # 设置刻度标签的字号大小
    axs.tick_params(axis='both', which='major', labelsize=28)  # 'both' 表示同时设置横纵坐标，12 为字号大小
    
    # plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    jpg_name = os.path.splitext(filename)[0]+'_20260204.jpg' ##_rmv_B7#####################################################################
    jpg_path = os.path.join('E:/LLL/01C/Program/NN/scatter_plots',jpg_name)
    plt.savefig(jpg_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

data_dir = 'E:/LLL/01C/Data/GEE_S2-20250406/merge/'
filename = 'S2_matchUp_2.5d_reviseName_QC_good.csv'
p1_train_data = TabularDataset(os.path.join(data_dir, filename))  


# Split the dataset into training and validation sets
train_data, val_data = train_test_split(p1_train_data, test_size=0.3, random_state=42)

# train_data = pd.concat([p1_train_data], axis=0, ignore_index=True)
train_autogluon_final(train_data, val_data)