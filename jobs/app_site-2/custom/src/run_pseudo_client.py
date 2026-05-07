



"""
This pseudo-client is responsible for transferring model weights through SURFdrive.

Main workflow:
1. Check whether the local model from the real training site has been uploaded to SURFdrive.
   - If the model file is not available yet, wait for 30 seconds and check again.
   - Repeat this process until the file becomes available.

2. Once the model file is available, download it from SURFdrive and save it locally.

3. Load the downloaded model weights and send them to the NVFLARE server for aggregation.
"""





import torch
import os 
import subprocess
import time
import nvflare.client as flare
import yaml



yaml_file=".../fed_pedadb/jobs/exp_set.yaml"

def load_read_params(model_path):
    # Load the model state_dict
    print(f"Loading model from {model_path}")
    params = torch.load(model_path,weights_only=True)  
    return params



##first check if the real site model is uploaded  surfdrive and then download
def check_file_there(file_name):
    password = "xxxxx"
    command=[
        "curl",
        "-u", f"xxxxxx@prinsesmaximacentrum.nl:{password}",
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
    password = "xxxxx"
    command=[
        "curl",
        "-u", f"xxxxxx@prinsesmaximacentrum.nl:{password}",
        "-o", saveto,
        f"https://surfdrive.surf.nl/files/remote.php/webdav/FL_Models/{file_name}"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Exit code:", result.returncode)




def main(model_local_folder='./models'):


    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    print(f'Client name: {client_name},h/n/n/n/n')





    while flare.is_running():
        input_model = flare.receive()


        with open(yaml_file, 'r') as f:
            exp_set = yaml.safe_load(f)
        expe_name=exp_set['experiment_name']
        round=exp_set['current_round']
        max_round=exp_set['total_rounds']+1

        if round==max_round:
            print("Training finished, exit")
            flare.exit()
            break
        else:

            print(f"current_round=nvflare: {input_model.current_round}, continue training:{round}")


            client_name='sitehd'

            while not check_file_there(f"{client_name}_{round}_fl.pth"):
                print(time.ctime(),f"File {client_name}_{round}_fl.pth not there yet, wait for 30 seconds")
                time.sleep(30) 
            
            print("File is there, start download")
            saveto1=f"{model_local_folder}/site_model/{expe_name}/{client_name}_{round}_fl.pth"
            os.makedirs(os.path.dirname(saveto1), exist_ok=True)

            pre_site_model=f"{model_local_folder}/site_model/{expe_name}/{client_name}_{round-2}_fl.pth"
            if os.path.exists(pre_site_model):
                os.remove(pre_site_model)


            download_surfdrive(f'{client_name}_{int(round)}_fl.pth',saveto1)
            params = load_read_params(saveto1)
            print("Finished Training")
        

            output_model = flare.FLModel(
                params=params,
                meta={'NUM_STEPS_CURRENT_ROUND':171},
            )

            flare.send(output_model)


if __name__ == "__main__":
    model_local_folder='./models'
    main(model_local_folder)
