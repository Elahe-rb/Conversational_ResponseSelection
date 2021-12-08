from config_params import dataset
#Dataset path.
FT_data={
    'ubuntu': '../../../../data/BERT/ubuntu_data/', #for colab: '../../../../data/BERT/ubuntu_data/ubuntu_dataset_1M.pkl',   #for mylap 'ubuntu': 'ubuntu_data/ubuntu_dataset_1M.pkl'
    'douban': 'douban_data/douban_dataset_1M.pkl',
    'e_commerce': 'e_commerce_data/e_commerce_dataset_1M.pkl'
}

data_file_path = FT_data[dataset]
save_model_path = FT_data[dataset] + "outputs/"