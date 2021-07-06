import os

'''
Holds all the paths relevant for loading data
'''

pre_path = os.path.abspath(os.path.join(".."))
raw_data_path = os.path.abspath(os.path.join(pre_path, "raw_data"))
data_path = os.path.abspath(os.path.join(pre_path, "data"))
serialized_path = os.path.abspath(os.path.join(data_path, "serialized"))

