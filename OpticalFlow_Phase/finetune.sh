


# python finetune.py --name viper-finetune --output results/viper/gma --restore_ckpt ./checkpoints/viper-finetune_loss_unsup.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision

# python finetune.py --name viper-finetune --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss3_alone_from_scratch_e1_bw_8000.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision --bwflow


# python evaluation.py --restore_ckpt ./checkpoints/pretrained_sintel.pth


# python finetune.py --name viper-finetune_loss_seg_unsup_e3 --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss_seg_unsup_e2.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision
# python finetune.py --name viper-finetune_loss_seg_unsup_e4 --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss_seg_unsup_e3.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision


# python finetune.py --name viper-finetune_loss_adj_seg_unsup_e3 --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss_adj_seg_unsup_e2.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision
python finetune.py --name viper-finetune_loss_seg_unsup_e1 --output results/viper/gma --restore_ckpt ./checkpoints/pretrained_sintel.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision
python finetune.py --name viper-finetune_loss_seg_unsup_e2 --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss_seg_unsup_e1.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision
python finetune.py --name viper-finetune_loss_seg_unsup_e3 --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss_seg_unsup_e2.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision
python finetune.py --name viper-finetune_loss_seg_unsup_e4 --output results/viper/gma --restore_ckpt ./results/viper/gma/viper-finetune_loss_seg_unsup_e3.pth --num_steps 25000 --lr 0.000125 --image_size 256 512 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision
