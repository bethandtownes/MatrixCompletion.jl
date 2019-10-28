
from typing import List
import re
import os 
import matplotlib.pyplot as plt

SIMULATION_DIRECTORY = '/home/jasonsun0310/.julia/dev/MatrixCompletion/test/test_result/'

def valid_dict_entry(s:str) -> bool:
    return s.find('=>') != -1

def clean_dict_entry(s:str) -> str:
    return s.replace(' ', '').replace('"','')

def convert_dict_value(lst:List[str]): 
    try:
        return lst[0], float(lst[1])
    except:
        return lst[0], lst[1]
    
def extract_rank_from_file_name(filename:str) -> int:
    return int(filename.replace('rank','').replace('.log',''))

def plot_list_of_pair(lst:List, title = '', legend = '', xlabel = '', ylabel = ''):
    fst_axis = list(map(lambda x:x[0], lst))
    snd_axis = list(map(lambda x:x[1], lst))
    plt.plot(fst_axis, snd_axis)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def locate_paragraph(s:str, start_token:str, end_token:str) -> List[str]:
    st = re.search(start_token, s).end()
    ed = re.search(end_token, s).start()
    return s[st:ed].split('\n')


def load_diagnostics_of_only_missing(s:str):
    dict_str = locate_paragraph(s, '(Only Missing)', 'Missing && Observed')
    transformed_dict = list(map(clean_dict_entry,filter(valid_dict_entry, dict_str)))
    transformed_dict = list(map(lambda x:x.split('=>'), transformed_dict))
    transformed_dict = list(map(convert_dict_value, transformed_dict))    
    return dict(transformed_dict)


def glob_diagnostics_of_only_missing(dir_name:str):
    os.chdir(SIMULATION_DIRECTORY + dir_name)
    log = dict()
    for filename in os.listdir():
        with open(filename) as f:
            log[filename] = load_diagnostics_of_only_missing(f.read())
    return log






# log = dict()
# for filename in os.listdir():
#     with open(filename) as f:
#         log_data = f.read()
#         dict_str = locate_paragraph(log_data, '(Only Missing)', 'Missing && Observed')
#         transformed_dict = list(map(clean_dict_entry,filter(valid_dict_entry, dict_str)))
#         transformed_dict = list(map(lambda x:x.split('=>'), transformed_dict))
#         transformed_dict = list(map(convert_dict_value, transformed_dict))
#         log[extract_rank_from_file_name(filename)] = dict(transformed_dict)

    

    
    
