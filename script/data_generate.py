from Botfip.datagen.opt_datagen import Op_dataset
from omegaconf import OmegaConf

if __name__ == '__main__':


    hyperparameters_yaml_path = '/home/cth/nfsroot/Botfip/configs/model_hyper.yaml'
    config = OmegaConf.load(hyperparameters_yaml_path)

    op_dataset = Op_dataset(hyperparameters_yaml_path,dataset_key = 'op_dataset_config')
    op_dataset.generate_formula_skeleton_dataset(op_dataset.config.dataset_path,ascend_ind = 1,if_ignore_root = True)
    op_dataset.generate_func_image_csv(op_dataset.config.dataset_path)

    #op_dataset.load_dataset(config.op_dataset_config.dataset_path)
    op_dataset.generate_function_image_dataset_multiprocess(op_dataset.config.dataset_path,op_dataset.config.chunk_size)

    #op_dataset.load_formula_skeleton_dataset(test_path)
    #op_dataset.load_dataset(test_path)
    #print(op_dataset.function_image_dataset)
