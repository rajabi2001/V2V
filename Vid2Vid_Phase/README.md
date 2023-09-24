# Vid2Vid_Phase

**Data Preparation**

Viper dataset is available via [Recycle-GAN](https://github.com/aayushbansal/Recycle-GAN/), and CityScapes sequence dataset (leftImg8bit_sequence_trainvaltest) is available [Here](https://www.cityscapes-dataset.com/downloads/). Please be advised that preparing datasets in triplets (as in Recycle-GAN) is not neccessary unless you also need to run Recycle-GAN.

Organize the dataset in such a way that it contains train/val set and source domain A/ target domain B hierarchically. For Viper-to-CityScapes experiments, A/B will be the frames from Viper/CityScapes while for Video-to-Label experiments, A/B will be the frames/label maps in Viper. 
```
path/to/data/
|-- train
|   |-- A
|   `-- B
`-- val
    |-- A
    `-- B
```

**Viper-to-CityScapes Experiment**
```
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 542 --loadSizeH 286 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 10 --lambda_unsup_cycle_A 10 --lambda_unsup_cycle_B 10 --lambda_cycle_A 0 --lambda_cycle_B 0 --lambda_content_A 1 --lambda_content_B 1 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 2
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 512 --loadSizeH 256 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 2
```

**Video-to-Label Experiment**

```
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2l_experiment --loadSizeW 286 --loadSizeH 286 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 0 --lambda_unsup_cycle_A 0 --lambda_unsup_cycle_B 10 --lambda_cycle_A 10 --lambda_cycle_B 10 --lambda_content_A 0 --lambda_content_B 0 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 5
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2l_experiment --loadSizeW 256 --loadSizeH 256 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 5
```
   
**Pretrained Models**

Pretrained models in both experiments are available [here]().

**Evaluation**

<!-- Pretrained FCN model is available [here](https://drive.google.com/file/d/1NmeC32gGoKqitxBax21-FaAKL_cBwwqg/view?usp=sharing). Please place it under .saved_models/.
```
python util/eval_rgb2lbl.py --exp_name path/to/test_output/images/\*fake_B.\* --map_cache_dir seg_map_cache
python util/eval_lbl2rgb.py --exp_name path/to/test_output/images/\*fake_A\* --pred_cache_dir pred_map_cache --mean False --model_path saved_models/fcn_model.pt
``` -->
             
**Acknowledgment**

