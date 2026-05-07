
"""
This is the main script for running training on real client 2, on machine 2,
without requiring a direct dependency on NVFLARE.

For each federated learning round, the script follows this workflow:

1. Check whether the global model from pseudo-client 2 has been uploaded to SURFdrive.
2. If the global model is not available yet, wait and check again.
3. Once the global model is available, download it from SURFdrive.
4. Use the downloaded global model to initialize local nnU-Net training.
5. Train the model locally on the client 2 dataset.
6. Save the locally trained model.
7. Upload the updated local model to SURFdrive so it can be collected by the pseudo-client on machine 1
   and sent back to the NVFLARE server for aggregation.
"""

import os 

os.environ['nnUNet_raw'] = ".../site-2/task/nnUNet/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = ".../site-2/task/nnUNet/nnUNet_preprocessed"
os.environ['nnUNet_results'] = ".../site-2/task/nnUNet/nnUNet_results"

yaml_file=".../site-2/job_real_site-2/exp_set.yaml"


from typing import Union
import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from torch.backends import cudnn
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnUNetTrainer_fl_aug_1000epochs import nnUNetTrainer_fl_aug_1000epochs as nnUNetTrainer_fl
import subprocess
import time
import yaml




def load_read_params(model_path):
    # Load the model state_dict
    params = torch.load(model_path)  # This already returns a state_dict
    return params


def train(model_input,model_output,dataset_name_or_id: Union[str, int],
                    configuration: str, fold: Union[int, str],
                    plans_identifier: str = 'nnUNetPlans',
                    num_gpus: int = 1,
                    export_validation_probabilities: bool = False,
                    continue_training: bool = False,
                    only_run_validation: bool = False,
                    disable_checkpointing: bool = False,
                    val_with_best: bool = False,
                    device: torch.device = torch.device('cuda')):


    nnUNet_preprocessed=os.environ['nnUNet_preprocessed']

    def get_trainer_from_args(model_input, model_output,dataset_name_or_id: Union[int, str],
                            configuration: str,
                            fold: int,
                            plans_identifier: str = 'nnUNetPlans',
                            device: torch.device = torch.device('cuda')):

        if dataset_name_or_id.startswith('Dataset'):
            pass
        else:
            try:
                dataset_name_or_id = int(dataset_name_or_id)
            except ValueError:
                raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                                f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                                f'input: {dataset_name_or_id}')


        # initialize nnunet trainer
        print("nnUNet_preprocessed",nnUNet_preprocessed)
        preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))



        plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
        print("plans_file",plans_file)
        plans = load_json(plans_file)
        dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
        nnunet_trainer = nnUNetTrainer_fl(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device,exp_set_yaml_file=yaml_file)
        


        return nnunet_trainer


    def run_training(model_input,model_output,dataset_name_or_id: Union[str, int],
                    configuration: str, fold: Union[int, str],
                    plans_identifier: str = 'nnUNetPlans',
                    num_gpus: int = 1,
                    export_validation_probabilities: bool = False,
                    continue_training: bool = False,
                    only_run_validation: bool = False,
                    disable_checkpointing: bool = False,
                    val_with_best: bool = False,
                    device: torch.device = torch.device('cuda')):
        if isinstance(fold, str):
            if fold != 'all':
                try:
                    fold = int(fold)
                except ValueError as e:
                    print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                    raise e

        if val_with_best:
            assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

        if num_gpus > 1:
            print("num_gpus",num_gpus)
            
        else:
            nnunet_trainer = get_trainer_from_args(model_input,model_output,dataset_name_or_id, configuration, fold,
                                                plans_identifier, device=device)

            if disable_checkpointing:
                nnunet_trainer.disable_checkpointing = disable_checkpointing

            assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        
            if torch.cuda.is_available():
                cudnn.deterministic = False
                cudnn.benchmark = True

            if not only_run_validation:
                nnunet_trainer.run_training(model_input,model_output)

        # if val_with_best:
            #    nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        # nnunet_trainer.perform_actual_validation(export_validation_probabilities)
            print("train_finish")


    run_training(model_input=model_input,model_output=model_output,plans_identifier=plans_identifier,dataset_name_or_id=dataset_name_or_id, configuration=configuration, fold=fold, device=device)


#############
#Surfdrive related functions
##########
def check_file_there(file_name):
    password = "xxxxxx"
    command=[
        "curl",
        "-u", f"xxx@prinsesmaximacentrum.nl:{password}",
        "-X", "PROPFIND",
        "-H", "Depth: 1",
        "https://surfdrive.surf.nl/files/remote.php/webdav/FL_Models/"
    ]

    # Run the command and filter output for 'check_poin'
    result = subprocess.run(command, capture_output=True, text=True)
    exsit=file_name in result.stdout
    if exsit:
        print(f"File {file_name} is there")
    return exsit



def download_surfdrive(file_name,saveto):
    password = "xxxxxx"
    command = [
        "curl",
        "-u", f"M.Ding-2@prinsesmaximacentrum.nl:{password}",
        "-o", saveto,
        f"https://surfdrive.surf.nl/files/remote.php/webdav/FL_Models/{file_name}"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Exit code:", result.returncode)



def upload_surfdrive(source,file_name):
    password = "xxxxxx"
    command = [
        "curl",
        "-u", f"xxx@prinsesmaximacentrum.nl:{password}",
        "-T", source,
        f"https://surfdrive.surf.nl/files/remote.php/webdav/FL_Models/{file_name}"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Exit code:", result.returncode)



#############

##########
def main( fold=0,
         dataset_name_or_id='Dataset105_HDPedRevised',
         configuration='3d_fullres',
         nnUNetPlans='PlansCombined',
         model_local_folder='./models',
         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    print('Start_training')
    with open(yaml_file, 'r') as f:
        exp_set = yaml.safe_load(f)
    max_rounds=exp_set['total_rounds']+1
    current_round=exp_set['current_round']
    init_model_path=exp_set['init_model_path']
    exper_name=exp_set['experiment_name']

    
    for round in range(current_round,max_rounds):
        if round == 0:
            print(f'Start round {round}, use provided_start model {init_model_path}')

            input_model_params = torch.load(init_model_path,weights_only=True)


            #reset the current round to 0 when start the first round
            with open(yaml_file, 'r') as f:
                exp_set = yaml.safe_load(f)
                exp_set['current_round']=0
            with open(yaml_file, 'w') as f:
                yaml.dump(exp_set, f)

        else:
            gl_model_path=f"{model_local_folder}/global_model/{exper_name}/gl_{round}.pth"
            os.makedirs(os.path.dirname(gl_model_path),exist_ok=True)


            while not check_file_there(f'gl_{round}.pth'):
                print(time.ctime(),f"File  gl_{round}.pth not there yet, wait for 30 seconds")
                time.sleep(30) 
            
            print("File is there, start download")

            download_surfdrive(f'gl_{round}.pth',gl_model_path)
            input_model_file=gl_model_path
            
            input_model_params = torch.load(input_model_file,weights_only=True)
        if round !=max_rounds-1:


            local_model_path=f"{model_local_folder}/local_model/{exper_name}/sitehd_{round}_fl.pth"
            os.makedirs(os.path.dirname(local_model_path),exist_ok=True)
            print("local_model_path",local_model_path)
        
            

            train(model_input=input_model_params,model_output=local_model_path,plans_identifier=nnUNetPlans,dataset_name_or_id=dataset_name_or_id, configuration=configuration, fold=fold, device=device)

            print("Finished Training")

            upload_surfdrive(local_model_path,f'sitehd_{round}_fl.pth')
            if round>0 and round<max_rounds-2:
                pre_local_model_path=f"{model_local_folder}/local_model/{exper_name}/sitehd_{round-1}_fl.pth"
                if os.path.exists(pre_local_model_path):
                    os.remove(pre_local_model_path)
                pre_gl_model_path=f"{model_local_folder}/global_model/{exper_name}/gl_{round-1}.pth"
                if os.path.exists(pre_gl_model_path):
                    os.remove(pre_gl_model_path)
              
          
            print(f"Upload {local_model_path} to surfdrive")
            
            with open(yaml_file, 'r') as f:
                exp_set = yaml.safe_load(f)
                exp_set['current_round']=round+1
            with open(yaml_file, 'w') as f:
                yaml.dump(exp_set, f)

    print("All rounds finished, exit")

        



if __name__ == "__main__":
    fold=0
    dataset_name_or_id='Dataset105_HDPedRevised'
    configuration='3d_fullres'
    nnUNetPlans='PlansCombined'
    model_local_folder='./models'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(fold=fold,
         dataset_name_or_id=dataset_name_or_id,
         configuration=configuration,
         nnUNetPlans=nnUNetPlans,
         model_local_folder=model_local_folder,
         device=device)