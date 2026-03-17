## Downloading Testset
### Blurry images and Raw events
Testset input:
[CodaLab_downloading_link](https://codalab.lisn.upsaclay.fr/my/datasets/download/3ed362b8-9084-414d-a5f3-d906708773cf); [Kaggle_downloading_link](https://www.kaggle.com/datasets/lei0331/highrev-testset)

The structure of the HighREV dataset with raw events is as following:

```
    --HighREV
    |----test
         |----blur
         |    |----SEQNAME_%5d.png
         |    |----...
         |----event
              |----SEQNAME_%5d_%2d.npz
              |----...

```
For each blurry image, there are several NPZ files containing events. By concatenating them, the events for the entire exposure time can be obtained. More details please refer to `./basicsr/data/npz_image_dataset.py`


```
git clone https://github.com/NikonD850/NTIRE26_event_deblur.git
cd NTIRE26_event_deblur

conda create -n iscas python=3.10 -y
conda activate iscas
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/state-spaces/mamba.git
cd mamba
python -m pip install . --no-build-isolation
cd ..

pip install -r requirements.txt
pip install timm

# Recommended editable install for modern pip/setuptools.
# Optional CUDA ops are skipped automatically when their sources are unavailable.
pip install -e .

# Legacy fallback:
# python setup.py develop --no_cuda_ext
```

## Prepare test data
```
# Create 21 channel voxel
bash scripts/ISCAS_Optics_build_voxel21_test.sh # Change your root of data in .sh

# Copy blur to sharp
cd 'your data root'
cp -r blur sharp
```

## Inference
```
cd NTIRE26_event_deblur
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/inference/2_ISCAS_Optics_1.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 basicsr/test.py -opt options/inference/2_ISCAS_Optics_2.yml --launcher pytorch
```

## Merge result
```
python scripts/fuse_png_from_two_roots.py
```
Then you can find your result in results/2_ISCAS_Optics_1_net_g_200000_tta_all8_0.5_2_ISCAS_Optics_1_net_g_80000_tta_all8_0.5
## Acknowledge
This repo is based on [EVSSM](https://github.com/kkkls/EVSSM) [ADHINet](https://github.com/wyang-vis/AHDINet) and [challenge official repo](https://github.com/AHupuJR/NTIRE2025_EventDeblur_challenge).
