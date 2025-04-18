file_path = '../data/r_p_e_up_depth/episode_10.hdf5'

import h5py

def print_structure(name, obj):
    indent = '    ' * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name} (Group)")
    else:
        print(f"{indent}{name} (Dataset)，shape = {obj.shape}")
    
    # 打印属性
    if obj.attrs:
        print(f"{indent}    Attributes:")
        for attr_name, attr_value in obj.attrs.items():
            print(f"{indent}        {attr_name}: {attr_value}")

# 打开文件
with h5py.File(file_path, 'r') as f:
    # 打印属性
    if f.attrs:
        print("Attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"{attr_name}: {attr_value}")
    # 遍历所有组和数据集
    f.visititems(print_structure)

