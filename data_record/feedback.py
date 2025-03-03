#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np

def feedback(df,save_position): 
    cont = 3
    cate_zinc = 7
    cate_ligand = 17

    dic_zinc = {0: 'ZnCl2', 1: 'Zn(NO3)2·6H2O', 2: 'Zn(CH3COO)2·2H2O', 3: 'ZnSO4',
        4: 'Zn(H2PO4)2·2H2O', 5: 'Zn(C3H5O3)2', 6: 'ZnBr2'}
    dic_ligand = {0: '咪唑', 1: '2-甲基咪唑', 2: '2-乙基咪唑', 3: '2-丙基咪唑',
        4: '4-甲基咪唑', 5: '2,4-二甲基咪唑', 6: '4,5-二甲基咪唑', 
        7: '2,4,5-三甲基咪唑', 8: '咪唑-2-甲醇', 9: '1-甲氧基甲基咪唑',
        10:'嘌呤',11:'组胺',12:'组氨酸',13:'2-咪唑烷酮',
        14:'1H-咪唑-2(3H)-硫酮',15:'噻唑',16:'2-甲基噻唑'}
        
    range_zinc = {'ZnCl2': [2.0000,10.0000], 'Zn(NO3)2·6H2O': [0.010,10.0000], 'Zn(CH3COO)2·2H2O': [0.010,0.8000], 
        'ZnSO4': [0.010,4.0000], 'Zn(H2PO4)2·2H2O': [0.010,0.8000], 
        'Zn(C3H5O3)2': [0.010,0.1200], 'ZnBr2':[0.010,10.0000]}
    range_ligand = {'咪唑': [0.010,10.000], '2-甲基咪唑': [0.010,4.000], '2-乙基咪唑': [0.010,10.000], 
        '2-丙基咪唑':[0.010,5.000], '4-甲基咪唑':[0.010,10.000], '2,4-二甲基咪唑':[0.010,10.000], '4,5-二甲基咪唑':[0.010,1.000],
        '2,4,5-三甲基咪唑':[0.010,10.000], '咪唑-2-甲醇':[0.010,10.000], '1-甲氧基甲基咪唑':[0.010,9.810],
        '嘌呤':[0.01,1.25],'组胺':[0.01,0.25],'组氨酸':[0.010,0.25],
        '2-咪唑烷酮':[0.010,10],'1H-咪唑-2(3H)-硫酮':[0.010,0.6],
        '噻唑':[0.010,14.090],'2-甲基噻唑':[0.010,11.090]}

    zinc_index_onehot = [_ for _ in range(cont,cont+cate_zinc)]
    ligand_index_onehot = [_ for _ in range(cont+cate_zinc,cont+cate_zinc+cate_ligand)]
    df['zinc_index_number'] = df[zinc_index_onehot].apply(lambda x: [i for i, v in enumerate(x) if v == 1], axis=1)
    df['zinc_index_number'] = df['zinc_index_number'].apply(lambda x: int(x[0])).map(dic_zinc)
    df['zinc_range_low'] = df['zinc_index_number'].map(range_zinc).apply(lambda x: x[0])
    df['zinc_range_up'] = df['zinc_index_number'].map(range_zinc).apply(lambda x: x[1])
    df['zinc_value'] = df[0] * (df['zinc_range_up'] - df['zinc_range_low']) + df['zinc_range_low']
    df['ligand_index_number'] = df[ligand_index_onehot].apply(lambda x: [i for i, v in enumerate(x) if v == 1], axis=1)
    df['ligand_index_number'] = df['ligand_index_number'].apply(lambda x: int(x[0])).map(dic_ligand)
    df['ligand_range_low'] = df['ligand_index_number'].map(range_ligand).apply(lambda x: x[0])
    df['ligand_range_up'] = df['ligand_index_number'].map(range_ligand).apply(lambda x: x[1])
    df['ligand_value'] = df[1] * (df['ligand_range_up'] - df['ligand_range_low']) + df['ligand_range_low']
    df['time'] = round((df[cont-1] * 90 + 30)/5)*5 

    df = df.drop(columns = [_ for _ in range(0,cont+cate_zinc+cate_ligand)])
    df = df.drop(columns = ['zinc_range_low','zinc_range_up','ligand_range_low','ligand_range_up'])
    df.to_excel(save_position, index=False)

    return 0
