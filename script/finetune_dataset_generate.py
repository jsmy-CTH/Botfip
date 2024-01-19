from datagen.opt_datagen import Op_dataset
from omegaconf import OmegaConf

if __name__ == '__main__':

    hyperparameters_yaml_path = '/home/cth/nfsroot/Botfip/configs/model_hyper.yaml'
    config = OmegaConf.load(hyperparameters_yaml_path)
    finetune_config = config.op_finetune_dataset_config

    op_dataset = Op_dataset(hyperparameters_yaml_path,dataset_key = 'op_finetune_dataset_config')
    op_dataset.load_dataset(config.op_finetune_dataset_config.dataset_path)
    print(len(op_dataset))
    #op_dataset.generate_func_image_csv(op_dataset.config.dataset_path)
    #op_dataset.generate_function_image_dataset_multiprocess(op_dataset.config.dataset_path,
    #                                                        op_dataset.config.chunk_size)

    #op_dataset.generate_finetune_formula_dataset(finetune_config)

    #finetune_dataset = Op_dataset(hyperparameters_yaml_path)