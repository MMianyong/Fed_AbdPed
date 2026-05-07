





"""
This is the main script to run nnU-Net training in a federated learning setting.

For each federated learning round, the script performs the following steps:
1. Loads the current global model.
2. Trains the model locally on the client data.
3. Saves the locally trained model.
4. Uploads the local model weights, for example via SURFdrive using `curl`,
   so they can be sent back to the server for aggregation.

Before running this script, the nnU-Net training entry point should be modified:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/run/run_training.py

The modified training script is then integrated into the NVFLARE workflow, so that
nnU-Net training is executed locally on each client during every federated learning round.

Before training, the dataset must be prepared and preprocessed in the nnU-Net format,
including the correct dataset name, plans name, and configuration.

In this setup, nnU-Net is trained using the `3d_fullres` configuration for 1000 epochs.
Therefore, the dataset must also be preprocessed for the `3d_fullres` configuration.
"""





import torch
import os 
import subprocess
import yaml


# Set environment variables 
os.environ['nnUNet_raw']="...nnUNet/nnUNet_raw"
os.environ['nnUNet_preprocessed'] ="...nnUNet/nnUNet_preprocessed"
os.environ['nnUNet_results']="...nnUNet/nnUNet_results"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


yaml_file=".../fed_pedadb/jobs/exp_set.yaml"


from typing import Union
from torch.backends import cudnn
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from src.nnUNetTrainer_fl_aug_1000epochs import nnUNetTrainer_fl_aug_1000epochs as nnUNetTrainer_fl

# NVFLARE client imports
import nvflare.client as flare



def load_read_params(model_path):
    # Load the model state_dict
    params = torch.load(model_path,weights_only=True)  # This already returns a state_dict
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




    nnUNet_preprocessed = os.environ['nnUNet_preprocessed']


    def get_trainer_from_args(model_input, dataset_name_or_id: Union[int, str],
                            configuration: str,
                            fold: int,
                            plans_identifier: str = 'nnUNetPlans',
                            device: torch.device = torch.device('cuda')):
        # Load the nnU-Net trainer and validate inputs.


        # handle dataset input. If it's an ID we need to convert to int from string
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
        


        # Only single-GPU training is supported in this federated learning setup.
        if num_gpus > 1:
            raise NotImplementedError("Multi-GPU training is not supported in this federated learning setup.")

        else:
            nnunet_trainer = get_trainer_from_args(model_input, dataset_name_or_id, configuration, fold,
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









# SURFdrive WebDAV credentials used by curl to upload model files.
password='xxxxxx'
def upload_surfdrive(source,file_name):
    command = [
        "curl",
        "-u", f"xxxxx@prinsesmaximacentrum.nl:{password}",
        "-T", source,
        f"https://surfdrive.surf.nl/files/remote.php/webdav/FL_Models/{file_name}"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Exit code:", result.returncode)



# Main function: initializes NVFLARE and runs the federated learning training loop.
def main( fold=0,
         dataset_name_or_id='Dataset201_umcunew',
         configuration='3d_fullres',
         nnUNetPlans='PlansCombined',
         model_local_folder='./models',
         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    print(f'Client name: {client_name}')


    # For each round, receive and load the model from the server.
    while flare.is_running():
        input_model = flare.receive()


        with open(yaml_file, 'r') as f:
            exp_set = yaml.safe_load(f)
        exper_name=exp_set['experiment_name']
        round=exp_set['current_round']


        # we set the max round to total_rounds+1, because we want to save the model after the last round,
        max_round=exp_set['total_rounds']+1


        print(f"current_round=nvflare: {input_model.current_round}, continue training:{round}")



        # The nvflare can initilze the model with provide scripts for net calss or read the models weight from provided path 
        # Here we make the client read the model weight from the provided path
        # the initial model directly from the path specified in the config.


        if round == 0:
            print(F'Start round {round}, use provided_start model ')
            input_model_file = exp_set['init_model_path']
            input_model_params = torch.load(input_model_file,weights_only=True, map_location="cuda:0")

            #read yaml file

            with open(yaml_file, 'r') as f:
                exp_set = yaml.safe_load(f)
                exp_set['current_round']=0
            with open(yaml_file, 'w') as f:
                yaml.dump(exp_set, f)



    
        else:
            input_model_params = input_model.params

            gl_model_path=f"{model_local_folder}/output_fl/{exper_name}/gl_{round}.pth"
            os.makedirs(os.path.dirname(gl_model_path),exist_ok=True)

            # Save the received global model locally and upload it to SURFdrive.
            torch.save(input_model_params, gl_model_path)
            file_name=f"gl_{round}.pth"
            upload_surfdrive(gl_model_path,file_name)
            print(f"Upload {file_name} to surfdrive")




        if round != max_round:


            os.makedirs(model_local_folder, exist_ok=True)

            # Define the output path for the locally trained model.
            model_output = f'{model_local_folder}/site_model/{exper_name}/{client_name}_{round}.pth'

            os.makedirs(os.path.dirname(model_output),exist_ok=True)


            train(model_input=input_model_params,
                  model_output=model_output,
                  plans_identifier=nnUNetPlans,
                  dataset_name_or_id=dataset_name_or_id, 
                  configuration=configuration, 
                  fold=fold, device=device)


            # Remove the global and site models from two rounds ago to save disk space.
            pre_gl_model_path=f"{model_local_folder}/output_fl/{exper_name}/gl_{round-2}.pth"
            if os.path.exists(pre_gl_model_path):
                os.remove(pre_gl_model_path)



            pre_model_output = f'{model_local_folder}/site_model/{exper_name}/{client_name}_{round-2}.pth'
            if os.path.exists(pre_model_output):
                os.remove(pre_model_output)


            print("Finished Training")
            params = load_read_params(model_output)

            # Update the current round number in the YAML config.
            with open(yaml_file, 'r') as f:
                exp_set = yaml.safe_load(f)
                exp_set['current_round']=round+1
            with open(yaml_file, 'w') as f:
                yaml.dump(exp_set, f)


            
            # NUM_STEPS_CURRENT_ROUND is used by NVFLARE to weight the aggregation.
            # Adjust this value to match the number of training samples at this site.
            output_model = flare.FLModel(
                params=params,
                meta={'NUM_STEPS_CURRENT_ROUND':49},
            )

            flare.send(output_model)
        elif round == max_round:
            print(f"Round {round} finished, no more training")
            break
     


if __name__ == "__main__":


    # Define the nnU-Net training configuration.
    fold=0
    dataset_name_or_id='Dataset201_umcunew'
    configuration='3d_fullres'
    nnUNetPlans='PlansCombined'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Define the local folder for saving global and site models.
    model_local_folder='./models'

    main(fold=fold,
         dataset_name_or_id=dataset_name_or_id,
         configuration=configuration,
         nnUNetPlans=nnUNetPlans,
         model_local_folder=model_local_folder,
         device=device)
