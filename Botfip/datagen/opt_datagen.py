
from ..operation.operation_tree import *
from ..operation.operation_func import *
from datetime import datetime
from tqdm import tqdm
import  os
from multiprocessing import Pool, cpu_count
from functools import partial
from omegaconf import OmegaConf
from ..datagen.data_utils import multiscale_mesh_generate,funcimg_transform
import warnings
from ..common.utils import *
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

    






class Op_dataset(Dataset):
    def __init__(self,
                 hyperparameters_yaml_path,
                 if_load=True,
                 dataset_key = 'op_dataset_config',
                 load_dataset_path = None,
                 set_node_num = None,
                 if_val = False,
                 ):

        super().__init__()

        config = OmegaConf.load(hyperparameters_yaml_path)
        self.config = config[dataset_key]
        self.op_tree_config = config.operation_tree_config

        operation_yaml_path = self.op_tree_config.operation_config_path
        self.operation_registry =  OperationRegistrySet.from_config_yaml(hyperparameters_yaml_path)

        self.num_of_formulas_skeleton = self.config.num_of_formulas_skeleton
        self.num_of_constants_group = self.config.num_of_constants_group
        self.num_of_op_assign = self.config.num_of_op_assign
        self.max_region_num = self.config.max_region_num
        self.max_node_range = self.config.max_node_range
        self.max_var_range = self.config.max_var_range
        self.region_distance_range = self.config.region_distance_range
        self.region_tensor_shape = self.config.region_tensor_shape
        self.chunk_size = self.config.chunk_size
        self.multiscale = self.config.multiscale


        formula_skeleton_columns = ['set_node_num', 'formula_skeleton_str', 'constant_num', 'tree_op_seq']
        self.formula_skeleton_dataset_df = pd.DataFrame(columns=formula_skeleton_columns)
        function_image_dataset_columns = ['set_node_num', 'skeleton_index', 'const_array']
        self.function_image_dataset = pd.DataFrame(columns=function_image_dataset_columns)

        if not if_val:
            if load_dataset_path is not None:
                self.load_dataset(load_dataset_path,set_node_num)
                self.dataset_path = load_dataset_path
            elif self.config.dataset_path is not None and if_load:
                self.load_dataset(self.config.dataset_path,set_node_num)
                self.dataset_path = self.config.dataset_path
            else:
                raise ValueError('The dataset path is None')
        elif if_load:
            val_path = self.config.val_dataset_path
            if val_path is not None:
                self.load_dataset(val_path,set_node_num)
                self.dataset_path = val_path
            else:
                raise ValueError('The val dataset path is None')



    def generate_finetune_formula_dataset(self,finetune_config):
        save_path = finetune_config.finetune_dataset_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        nowtime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_name = 'formula_skeleton_dataset.csv'

        new_formula_skeleton =  self.formula_skeleton_dataset_df[self.formula_skeleton_dataset_df['set_node_num'] == finetune_config.finetune_set_node_num]
        new_formula_skeleton = new_formula_skeleton.sample(n = finetune_config.finetune_formula_num,replace = False)
        new_formula_skeleton.to_csv(os.path.join(save_path,save_name),index = False)

    def __len__(self):
        return len(self.function_image_dataset)

    def __getitem__(self, idx):
        assert len(self.function_image_dataset) != 0, "The dataset is empty"

        skeleton_index = self.function_image_dataset['skeleton_index'].iloc[idx]
        tree_seq_op = str2list(self.formula_skeleton_dataset_df['tree_op_seq'].iloc[skeleton_index])


        const_array = self.function_image_dataset['const_array'].iloc[idx]
        if pd.isna(const_array):
            const_array = None
        elif isinstance(const_array, str):
            const_array = str2list_float(const_array)

        set_node_num = self.function_image_dataset['set_node_num'].iloc[idx]
        img_index = self.function_image_dataset['img_index'].iloc[idx]

        img_path = os.path.join(self.dataset_path,'img',str(set_node_num), str(img_index) + '.npy')
        funcimg = np.load(img_path)

        new_funcimg,funcimg_max,funcimg_min = funcimg_transform(funcimg,
                                    img_range = self.config.img_compress_range,
                                    nan_replace = 0.,)

        funcimg[np.isnan(funcimg) | np.isinf(funcimg)] = 0

        return {
            'funcimg': torch.tensor(new_funcimg,dtype=torch.float32),
            'original_funcimg': torch.tensor(funcimg,dtype=torch.float32), # for visualization
            'funcimg_max': torch.tensor(funcimg_max,dtype=torch.float32),
            'funcimg_min': torch.tensor(funcimg_min,dtype=torch.float32),
            'opseq':(tree_seq_op, const_array),
            'const_array': const_array,
            'set_node_num': set_node_num,
            'img_index': img_index,
            'skeleton_index': skeleton_index,
        }


    def generate_formula_skeleton_dataset(self,formula_save_path_dir,ascend_ind = 3,if_ignore_root = True):
        if not os.path.exists(formula_save_path_dir):
            os.makedirs(formula_save_path_dir)
        save_name = 'formula_skeleton_dataset.csv'
        save_path = os.path.join(formula_save_path_dir, save_name)
        start_node = self.max_node_range[0]
        end_node = self.max_node_range[1]
        for set_node_num in range(start_node,end_node,ascend_ind):
            for _ in tqdm(range(self.num_of_formulas_skeleton)):
                op_tree = OperationRandomTree(num_nodes=set_node_num,
                                              config=self.op_tree_config,
                                              operation_registry_set=self.operation_registry,)
                for _ in range(self.num_of_op_assign):
                    op_tree.random_assign_operations()
                    sp_input =  op_tree.variable_symbols
                    func = op_tree.func_iteration(0)
                    try:
                        formula_skeleton_str = None
                        @time_out(5, timeout_callback)
                        def new_func(sp_input):
                            return str(func(sp_input,if_skeleton=True))
                        formula_skeleton_str = new_func(sp_input)
                        if 'zoo' in formula_skeleton_str or 'x' not in formula_skeleton_str:
                            print('The generated formula does not meet the requirements, pass')
                            continue
                        tree_op_seq, _ = op_tree.tree_serialized_encode_seq(if_ignore_root = if_ignore_root)
                        node_num = len(op_tree.node_info)
                        append_df = pd.DataFrame({
                            'set_node_num': node_num,
                            'formula_skeleton_str': formula_skeleton_str,
                            'constant_num': op_tree.constants_num,
                            'tree_op_seq': list2str(tree_op_seq),
                        }, index=[0])
                        self.formula_skeleton_dataset_df = pd.concat([self.formula_skeleton_dataset_df, append_df], ignore_index=True)
                    except:
                        print('sympy generation error, continue')
                        continue
            self.formula_skeleton_dataset_df.to_csv(save_path,index=False)
        self.formula_skeleton_dataset_df['original_index'] = self.formula_skeleton_dataset_df.index
        return self.formula_skeleton_dataset_df

    def load_formula_skeleton_dataset(self,formula_save_path_dir):
        save_name = 'formula_skeleton_dataset.csv'
        save_path = os.path.join(formula_save_path_dir, save_name)
        self.formula_skeleton_dataset_df = pd.read_csv(save_path)
        return self.formula_skeleton_dataset_df

    def load_dataset(self, formula_save_path_dir,set_node_num = None):
        csv_list = os.listdir(formula_save_path_dir)
        if 'formula_skeleton_dataset.csv' in csv_list:
            fs_dataset_path = os.path.join(formula_save_path_dir, 'formula_skeleton_dataset.csv')
            self.formula_skeleton_dataset_df = pd.read_csv(fs_dataset_path)
            #if set_node_num is not None:
            #    self.formula_skeleton_dataset_df = self.formula_skeleton_dataset_df[self.formula_skeleton_dataset_df['set_node_num'] == set_node_num]
            if 'img.csv' in csv_list:
                img_dataset_path = os.path.join(formula_save_path_dir, 'img.csv')
                self.function_image_dataset = pd.read_csv(img_dataset_path)
                if set_node_num is not None:
                    if isinstance(set_node_num,int):
                        self.function_image_dataset = self.function_image_dataset[self.function_image_dataset['set_node_num'] == set_node_num]
                    elif isinstance(set_node_num,list):
                        self.function_image_dataset = self.function_image_dataset[self.function_image_dataset['set_node_num'].isin(set_node_num)]
        else:
            raise Exception('formula_skeleton_dataset.csv not exist')
        return self.formula_skeleton_dataset_df



    def generate_function_image_dataset_multiprocess(self,img_save_dir,chunk_size):
        assert  len(self.formula_skeleton_dataset_df) > 0, 'formula_skeleton_dataset_df is empty'
        assert  len(self.function_image_dataset) > 0, 'function_image_dataset is empty'
        img_save_dir = os.path.join(img_save_dir,'img')
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        node_group = self.function_image_dataset.groupby('set_node_num')

        for set_node_num,img_group in node_group:
            node_img_save_dir = os.path.join(img_save_dir,str(set_node_num))
            if not os.path.exists(node_img_save_dir):
                os.makedirs(node_img_save_dir)

            chunks = [img_group.iloc[i:i + chunk_size] for i in range(0, len(img_group), chunk_size)]

            with Pool(processes=cpu_count()) as pool:
                pool.map(partial(generate_img,
                                 set_node_num,
                                 node_img_save_dir,
                                 self.formula_skeleton_dataset_df,
                                 self.operation_registry,
                                 self.op_tree_config,
                                self.config,),
                         chunks)



    def generate_func_image_csv(self,img_csv_save_dir,set_node = None,max_index = None):
        csv_save_dir = os.path.join(img_csv_save_dir,'img')
        if not os.path.exists(csv_save_dir):
            os.makedirs(csv_save_dir)

        self.formula_skeleton_dataset_df['original_index'] = self.formula_skeleton_dataset_df.index

        if set_node is not None:
            if isinstance(set_node,list):
                fs_dataset_df = self.formula_skeleton_dataset_df[self.formula_skeleton_dataset_df['set_node_num'].isin(set_node)].copy()
            else:
                fs_dataset_df = self.formula_skeleton_dataset_df[self.formula_skeleton_dataset_df['set_node_num'] == set_node].copy()
        else:
            fs_dataset_df = self.formula_skeleton_dataset_df.copy()

        fs_group = fs_dataset_df.groupby('set_node_num')
        for set_node_num,fs_group in fs_group:
            node_save_dir = os.path.join(csv_save_dir,str(set_node_num))
            if not os.path.exists(node_save_dir):
                os.makedirs(node_save_dir)
            for index, row in tqdm(fs_group.iterrows(), total=fs_group.shape[0]):
                output_df = pd.DataFrame(columns=self.function_image_dataset.columns)
                if row['constant_num'] == 0:
                    num_of_constants_group = 3
                else:
                    num_of_constants_group = self.num_of_constants_group

                for _ in range(num_of_constants_group):
                    const_array = np.random.uniform(low=self.op_tree_config ['constants_range'][0],
                                                    high=self.op_tree_config ['constants_range'][1],
                                                    size=row['constant_num'])
                    const_list = list2str(np.round(const_array, 2).tolist())
                    img_df = pd.DataFrame({
                        'set_node_num': set_node_num,
                        'skeleton_index': row['original_index'],
                        'const_array': const_list,
                    }, index=[0])
                    output_df = pd.concat([output_df, img_df], ignore_index=True)
                self.function_image_dataset = pd.concat([self.function_image_dataset, output_df], ignore_index=True)
                self.function_image_dataset['img_index'] = self.function_image_dataset.index
                self.function_image_dataset.to_csv(os.path.join(img_csv_save_dir,'img.csv'),index=False)
                if max_index is not None:
                    if len(self.function_image_dataset) > max_index:
                        return self.function_image_dataset
        return self.function_image_dataset


def generate_img(set_node_num,
                 node_img_save_dir,
                 formula_skeleton_dataset_df,
                 operation_registry,
                 op_tree_config,
                 dataset_config,
                 #region_tensor_shape,
                 img_df):

    op_tree = OperationRandomTree(num_nodes=set_node_num,
                                  config =  op_tree_config,
                                  operation_registry_set=operation_registry,)
    encoder_vector_dict = {
        'tree_op_seq': None,
        'const_array': None,
    }

    for _,row in tqdm(img_df.iterrows()):
        skeleton_index = row['skeleton_index']
        try:
            if not pd.isna(row['const_array']):
                const_array = np.array(str2list_float(row['const_array']))
            else:
                const_array = None
        except Exception as e:
            print('error:',row['const_array'])
            raise e
        img_index = row['img_index']
        tree_op_seq = str2list(formula_skeleton_dataset_df.iloc[skeleton_index]['tree_op_seq'])

        if encoder_vector_dict['tree_op_seq'] != tree_op_seq:
            encoder_vector_dict['tree_op_seq'] = tree_op_seq
            encoder_vector_dict['constants_array'] = const_array
            op_tree.load_tree(encoder_vector_dict,root_default_op = 'linear')
        else:
            op_tree.set_num_parameters(const_array)

        meshgrid,_ = multiscale_mesh_generate(dataset_config.multiscale,
                                            [-1,1],
                                            dataset_config.img_shape,
                                            op_tree_config.max_var_types)
        meshgrid = meshgrid.numpy()

        func_img = op_tree.formula_image(meshgrid)
        while len(func_img.shape) <3:
            func_img = np.expand_dims(func_img,0)
        #print(func_img)
        img_save_name = str(img_index) + '.npy'
        img_save_path = os.path.join(node_img_save_dir,img_save_name)
        np.save(img_save_path,func_img)
    print('generate function image dataset finished')









