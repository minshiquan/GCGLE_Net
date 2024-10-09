## Data structure

```bash

|--data
  |___Synapse
      |---test_vol_h5
      |   |--case0001.npy.h5
      |   |__*.npy.h5
      |___train_npz
          |--case0005_slice000.npz
          |__*.npz
```


- Train
```bash
python train.py --dataset Syanpse --root_path your DATA_DIR --max_epochs 400 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```
- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 400 --base_lr 0.05 --img_size 224 --batch_size 24
```
