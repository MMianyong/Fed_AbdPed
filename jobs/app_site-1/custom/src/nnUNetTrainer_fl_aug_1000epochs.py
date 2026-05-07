





"""
This script defines a custom nnU-Net trainer for federated learning with 1000 training epochs
and a custom polynomial learning-rate scheduler.

The trainer is based on `nnUNetTrainer`, with minimal modifications to support the federated
learning workflow and the custom `PolyLRScheduler`.

Main modifications:
1. Model initialization:
   - At the beginning of training, the trainer loads model weights from the current global model saved locally.

2. Optimizer and learning-rate scheduler:
   - The `configure_optimizers` method is modified to use the custom `PolyLRScheduler`
     instead of the default nnU-Net scheduler.
   - The custom scheduler reads the accumulated number of epochs across federated rounds and
     computes the learning rate based on the total number of epochs completed over all rounds.

3. Model saving:
   - After train end of each round, the new model weights are saved locally to be sent to the server for aggregation.

4. Data augmentation:
   - The data augmentation settings are modified to allow a wider range of rotation angles for lateral orientation of the data in this specific use case.

5. We need mofify it for single gpu training.

All other components remain mostly unchanged from the original `nnUNetTrainer`.
"""

import os 
import numpy as np
import shutil
import sys
import torch
from typing import Tuple, Union, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.utilities.file_and_folder_operations import join, isfile, save_json, maybe_mkdir_p
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform




# new imports for the custom trainer for federated learning with 1000 epochs and the custom poly lr scheduler
from src.spatial import SpatialTransform as SpatialTransform
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
import yaml
from src.adpt_polylr import PolyLRScheduler
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms


def save_model_locally(model, model_output_path):
    torch.save(model.state_dict(), model_output_path)






class nnUNetTrainer_fl_aug_1000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),exp_set_yaml_file: str = "./fed_pedabd/jobs/exp_set.yaml"):


        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # Save device separately (since the parent class doesn't use it)
        self.device = device
        self.local_rank = 0
        self.exp_set_yaml_file = exp_set_yaml_file

        with open(self.exp_set_yaml_file, 'r') as f:
            exp_set = yaml.safe_load(f)

        self.initial_lr = exp_set['initial_lr']
        self.num_epochs = exp_set['epoch_per_round']
       

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # call parent to get the defaults
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # now disable mirroring entirely
        mirror_axes = None
        # also disable it at inference time
        self.inference_allowed_mirroring_axes = None
        ##new_change ped
        dim=3
        patch_size = self.configuration_manager.patch_size
        #only change rotation in ax for adapt to the lateral orientation 
        rotation_for_DA_customized=(-120 / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                    rotation_for_DA_customized,
                                    rotation_for_DA,
                                    rotation_for_DA,
                                    (0.85, 1.25))
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

       
    #Add the model_input to read the model weights from the global model
    def initialize(self,model_input):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                    self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)


            self.network.load_state_dict(model_input, strict=True)
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer=optimizer, initial_lr=self.initial_lr, max_steps=self.num_epochs, yaml_file=self.exp_set_yaml_file)
        return optimizer, lr_scheduler




    # modify to make the data augmentaion of rotation can go beyond default -30 to 30
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        # Call the original method and get its list of transforms
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label
        ).transforms

        # Remove the original SpatialTransform
        new_transforms = [
            t for t in transforms
            if 'SpatialTransform' not in t.__class__.__name__
        ]

        # Re-add a custom SpatialTransform with modified behavior
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            patch_size_spatial = patch_size[1:]
            new_transforms.insert(0, Convert3DTo2DTransform())
            new_transforms.append(Convert2DTo3DTransform())
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        new_transforms=[SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.5,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )]+new_transforms
        

        return ComposeTransforms(new_transforms)




    #add the model output path to save the model locally after training is done for fl
    def on_train_end(self,model_output_path):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        #self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))

        save_model_locally(self.network, model_output_path)
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")




    def on_train_start(self,model_input):
        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference


        if not self.was_initialized:
            self.initialize(model_input)
  
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.local_rank == 0:
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)



        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def run_training(self,model_input,model_output_path):
        self.on_train_start(model_input)



        for epoch in range(self.current_epoch, self.num_epochs):
            if self.current_epoch ==0:

                #get the round information, and make validation
                with open(self.exp_set_yaml_file, 'r') as f:
                    exp_set = yaml.safe_load(f)

                round=exp_set['current_round']

                ####changed, make the validation every start of the run 
                self.print_to_log_file(f"Validate the global model of round {round}", also_print_to_console=True)
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):                  
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    outputs_collated = collate_outputs(val_outputs)
                    tp = np.sum(outputs_collated['tp_hard'], 0)
                    fp = np.sum(outputs_collated['fp_hard'], 0)
                    fn = np.sum(outputs_collated['fn_hard'], 0)

                    global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
                    mean_fg_dice = np.nanmean(global_dc_per_class)
                    loss_here = np.mean(outputs_collated['loss'])
                    self.print_to_log_file('mean_fg_dice', mean_fg_dice,also_print_to_console=True)
                    self.print_to_log_file('dice_per_class_or_region', global_dc_per_class, self.current_epoch,also_print_to_console=True)
                    self.print_to_log_file('val_losses', loss_here, also_print_to_console=True)

            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
   

            self.on_epoch_end()

        self.on_train_end(model_output_path)