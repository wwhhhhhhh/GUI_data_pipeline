import yaml
import subprocess
import moxing as mox


# mox.file.copy_parallel("s3://bucket-green-huadong2-711/01.USERS/w00878018/data/GUI_data/html_260227_processed/", "/cache/GUI_data/html_260227_processed/")

def prepare_data(yml_path, cache_dir):
    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 假设 data[-4][0] 是 parquet 路径, data[-4][1] 是 zip 路径
    for data_item in data['info']:
        s3_parquet_prefix = data_item[5][0]
        s3_zip_prefix = data_item[5][1]

        local_parquet_prefix = s3_parquet_prefix.replace('s3://', cache_dir)
        local_zip_prefix = s3_parquet_prefix.replace('s3://', cache_dir)

        mox.file.copy_parallel(s3_parquet_prefix, local_parquet_prefix)
        mox.file.copy_parallel(s3_zip_prefix, local_zip_prefix)
    
        # print(f'mox.file.copy_parallel("{s3_parquet_prefix}", "{local_parquet_prefix}")')
        # print(f'mox.file.copy_parallel("{s3_zip_prefix}", "{local_zip_prefix}")')



prepare_data('GUI_data/PANGU_T2I_L5_data.yml', '/cache/PANGU_T2I_L5_data')