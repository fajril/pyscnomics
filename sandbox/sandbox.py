import numpy as np


dict_1 = {"RC Bottom Limit": np.array([0, 1]),
          "RC Top Limit": np.array([1, 2]),
          "Pre Tax CTR Oil": np.array([0.3137260, 0.313233]),
          "Pre Tax CTR Gas": np.array([0.5498040, 0.121324]),
          }


dict_2 = {
    'condition_1': {'RC Bottom Limit': 0,
                    'RC Top Limit': 1,
                    'Pre Tax CTR Oil': 0.3137260,
                    'Pre Tax CTR Gas': 0.5498040,},

    'condition_2': {'RC Bottom Limit': 1,
                    'RC Top Limit': 2,
                    'Pre Tax CTR Oil': 0.313233,
                    'Pre Tax CTR Gas': 0.121324,}}

length_condition = len(dict_1[list(dict_1.keys())[0]])
enum_value = list(range(1, length_condition + 1))
dict_3 = {}
for index, step in enumerate(enum_value):
    dict_isi = {'bot_limit': dict_1['RC Bottom Limit'][index],
                'top_limit': dict_1['RC Top Limit'][index],
                'ctr_oil': dict_1['Pre Tax CTR Oil'][index],
                'ctr_gas': dict_1['Pre Tax CTR Gas'][index]}

    key_string = 'condition_' + str(step)
    dict_3[key_string] = dict_isi

print(dict_3)