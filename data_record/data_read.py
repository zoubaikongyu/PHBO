#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import torch 

def data_processing(data):
    data['key'] = data['①Zn源'].astype(str) + '|' + data['②Zn2+浓度 (M)'].astype(str) + '|' + data['③配体'].astype(str) + '|' + data['④配体浓度 (M)'].astype(str) + '|' + data['⑨反应时间 (min)'].astype(str)
    data = data.groupby('key').agg({'均值': 'mean'}).reset_index()
    data[['①Zn源', '②Zn2+浓度 (M)', '③配体', '④配体浓度 (M)', '⑨反应时间 (min)']] = data['key'].str.split('|', expand=True)
    data['②Zn2+浓度 (M)'] = data['②Zn2+浓度 (M)'].astype(float)
    data['④配体浓度 (M)'] = data['④配体浓度 (M)'].astype(float)
    data['⑨反应时间 (min)'] = data['⑨反应时间 (min)'].astype(float).astype(int)
    data = data.drop(columns=['key'])
    return data
    
def data_read(data_path):
       dic_zinc = {'ZnCl2': 0, 'Zn(NO3)2·6H2O': 1, 'Zn(CH3COO)2·2H2O': 2, 
                     'ZnSO4': 3, 'Zn(H2PO4)2·2H2O': 4, 
                     'Zn(C3H5O3)2': 5, 'ZnBr2':6}
       dic_ligand = {'咪唑': 0, '2-甲基咪唑': 1, '2-乙基咪唑': 2, 
                     '2-丙基咪唑':3, '4-甲基咪唑':4, '2,4-二甲基咪唑':5, '4,5-二甲基咪唑':6,
                     '2,4,5-三甲基咪唑':7, '咪唑-2-甲醇':8, '1-甲氧基甲基咪唑':9,
                     '嘌呤':10,'组胺':11,'组氨酸':12,'2-咪唑烷酮':13,
                     '1H-咪唑-2(3H)-硫酮':14,'噻唑':15,'2-甲基噻唑':16,}
       range_zinc = {'ZnCl2': [2.0000,10.0000], 'Zn(NO3)2·6H2O': [0.010,10.0000], 'Zn(CH3COO)2·2H2O': [0.010,0.8000], 
                     'ZnSO4': [0.010,4.0000], 'Zn(H2PO4)2·2H2O': [0.010,0.8000], 
                     'Zn(C3H5O3)2': [0.010,0.1200], 'ZnBr2':[0.010,10.0000]}
       range_ligand = {'咪唑': [0.010,10.000], '2-甲基咪唑': [0.010,4.000], '2-乙基咪唑': [0.010,10.000], 
                     '2-丙基咪唑':[0.010,5.000], '4-甲基咪唑':[0.010,10.000], '2,4-二甲基咪唑':[0.010,10.000], '4,5-二甲基咪唑':[0.010,1.000],
                     '2,4,5-三甲基咪唑':[0.010,10.000], '咪唑-2-甲醇':[0.010,10.000], '1-甲氧基甲基咪唑':[0.010,9.810],
                     '嘌呤':[0.01,1.25],'组胺':[0.01,0.25],'组氨酸':[0.010,0.25],
                     '2-咪唑烷酮':[0.010,10],'1H-咪唑-2(3H)-硫酮':[0.010,0.6],
                     '噻唑':[0.010,14.090],'2-甲基噻唑':[0.010,11.090]}
       range_reaction_time = [30,120]
       data = pd.read_excel(data_path)
       data = data_processing(data)
       zinc = data['①Zn源']
       zinc_value = data['②Zn2+浓度 (M)']
       norm_zinc_value = [0 for _ in range(len(zinc))]
       ligand = data['③配体']
       ligand_value = data['④配体浓度 (M)']
       norm_ligand_value = [0 for _ in range(len(zinc))]
       reaction_time = data['⑨反应时间 (min)']
       norm_reaction_time = [0 for _ in range(len(zinc))]
       one_hot_zinc = [[0 for _ in range(len(dic_zinc))] for _ in range(len(zinc))]
       one_hot_ligand = [[0 for _ in range(len(dic_ligand))] for _ in range(len(zinc))]
       cat_zinc = [dic_zinc[x] for x in zinc]
       cat_ligand = [dic_ligand[x] for x in ligand]

       for i in range(len(zinc)):
              norm_zinc_value[i] = (zinc_value[i] - range_zinc[zinc[i]][0]) / (range_zinc[zinc[i]][1]-range_zinc[zinc[i]][0])
              norm_ligand_value[i] = (ligand_value[i] - range_ligand[ligand[i]][0]) / (range_ligand[ligand[i]][1]-range_ligand[ligand[i]][0])
              norm_reaction_time[i] = (reaction_time[i] - range_reaction_time[0]) / (range_reaction_time[1]-range_reaction_time[0])
              one_hot_zinc[i][cat_zinc[i]] = 1
              one_hot_ligand[i][cat_ligand[i]] = 1
       X = [[a,b,c]+d+e for a,b,c,d,e, in zip(
              norm_zinc_value,norm_ligand_value,norm_reaction_time,one_hot_zinc,one_hot_ligand)] 
       X = torch.tensor(X)
       Y = torch.tensor([[a] for a in [b for b in data['均值']]])

       return X,Y

