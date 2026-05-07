# Job Definitions

This folder defines the federated learning tasks for the server and clients on Machine 1 and Machine 2.


The implementation is adapted from NVFLARE with client API and nnU-Net with minimal modifications.

The goal is to run nnU-Net training inside an NVFLARE federated learning workflow while keeping the code simple and easy to modify. Instead of relying on NVFLARE built-in training abstractions, we use the NVFLARE Client API and custom scripts to control model loading, local training, saving, and aggregation.

Reference: [NVFLARE Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html)

---

## Preparation

Before running the federated learning job, the dataset should be prepared and preprocessed in the nnU-Net format. The dataset name, plans name, and configuration must match the settings used in the client training script and NVFLARE configuration files. Some paths in the scripts also need to be adjusted manually, including the path to the nnU-Net environment, the `exp_set.yaml` file, and the directory used to save global models. In addition, the same initialized global model should be distributed to all clients before training starts. In this setup, model transfer through SURFdrive uses the [**WebDAV protocol**](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/126222571/RD+Uploading+files+to+a+Public+link), with password-based authentication required for uploading and downloading files.
---

## Job Folder Structure (machine 1)

```text
jobs/
├── app_server
│   ├── config
│   │   └── config_fed_server.json
│   └── custom
│       └── src
├── app_site-1
│   ├── config
│   │   └── config_fed_client.json
│   └── custom
│       └── src
│           ├── adpt_polylr.py
│           ├── nnUNetTrainer_fl_aug_1000epochs.py
│           ├── run_client_umcu_aug_1000epochs.py
│           ├── sample_scalar.py
│           └── spatial.py
├── app_site-2
│   ├── config
│   │   └── config_fed_client.json
│   └── custom
│       └── src
│           └── run_pseudo_client.py
├── exp_set.yaml
├── meta.json
└── README.md
```

---

## Configuration Files

### `config_fed_server.json`

This file defines the server-side federated learning configuration. The default NVFLARE controller (`nvflare.app_common.workflows.fedavg.FedAvg`).

The main parameters to configure:

- Number of clients
- Number of federated learning rounds
- Initial global model checkpoint path

In this setup, **200 federated learning rounds** are used, with `max_rounds` set to `201` to ensure the global model from round 200 is saved correctly.

The initial model checkpoint path:

```json
"source_ckpt_file_full_name": "/path/to/initial/global/model/checkpoint.pth"
```

This parameter defines the checkpoint used to initialize the global model before federated learning starts.

---


### `exp_set.yaml`

This file stores experiment-level settings shared across rounds.

It defines:

- Total number of epochs
- Epochs per federated learning round
- Initial model path
- Current federated learning round number
- Accumulated epoch count across previous rounds

The accumulated epoch count is important because the learning-rate scheduler uses it to compute the correct learning rate across all rounds. This file is updated after each round so the next round continues from the correct training state.

---

## Site 1: Real Training Client

`app_site-1` is the real client that performs local nnU-Net training.

### `config_fed_client.json`

This file defines the which script the client should run for training.

---

### `run_client_umcu_aug_1000epochs.py`

This is the main client-side training script.

It is modified from the [**original nnU-Net training**](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/run/run_training.py)and wraps the NVFLARE Client API.

Main responsibilities:

1. Receive or load the current global model.
2. Start local nnU-Net training with data from client utr.
3. Save the locally trained model.
4. Send the local model to server for federated average  aggregation.

---

### `nnUNetTrainer_fl_aug_1000epochs.py`

This file defines the custom nnU-Net trainer.

It extends the default `nnUNetTrainer` with minimal changes:

1. Load the model weights locally at the start of each federated learning round.
2. Save the locally trained model weights at the end of each round.
3. Use the customized polynomial learning-rate scheduler from `adpt_polylr.py`.
4. Apply modified data augmentation settings for this specific use case.

---

### `adpt_polylr.py`

This file defines the adapted polynomial learning-rate scheduler.

The scheduler computes the learning rate based on the total training progress across all federated rounds, so the schedule behaves as if training is continuous despite being split across multiple rounds.

---

### `spatial.py` and `sample_scalar.py`

These files modify the default nnU-Net data augmentation by disabling mirroring and extending the rotation range beyond the standard ±30° to better accommodate lateral/supine orientation differences in the dataset.

---

## Site 2: Pseudo-Client

`app_site-2` is a pseudo-client. It does not perform real nnU-Net training.

### `run_pseudo_client.py`

The pseudo-client is responsible for transferring model weights through SURFdrive.

Main workflow:

1. Upload the aggregated global model to SURFdrive.
2. Check whether the real client has uploaded its locally trained model.
3. If the local model is not available yet, wait and check again.
4. Download the real client's updated local model from SURFdrive.
5. Send the downloaded local model to the NVFLARE server for aggregation.

This pseudo-client bridges model transfer between the real training site and the NVFLARE server.

---

## Real Site 2 (`job_real_site-2/`)

This folder contains the scripts to be deployed on **Machine 2** (the real HEI client). It mirrors the training logic of `app_site-1` but runs standalone, outside NVFLARE, communicating with the server indirectly via SURFdrive.

```text
job_real_site-2/
├── adpt_polylr.py
├── exp_set.yaml
├── nnUNetTrainer_fl_aug_1000epochs.py
├── run_client_utr_fl_aug_1000epochs.py
└── spatial.py
```

### `run_client_utr_fl_aug_1000epochs.py`

The main entry point on Machine 2. It monitors SURFdrive for the latest global model, runs local nnU-Net training, and uploads the updated local model back to SURFdrive for the pseudo-client to collect.

The remaining files (`adpt_polylr.py`, `nnUNetTrainer_fl_aug_1000epochs.py`, `spatial.py`, `exp_set.yaml`) serve the same roles as their counterparts in `app_site-1/custom/src/`.
