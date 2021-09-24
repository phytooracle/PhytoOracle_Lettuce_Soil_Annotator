# PhytoOracle Lettuce Soil Annotator

## Train
image: `armanzarei/lettuce-soil-segmentation-train`
- `docker run -it --gpus all -v ${DATASET_PATH}:/src/data armanzarei/lettuce-soil-segmentation-train`
  - `--num_epochs` can be passed to control the number of train epochs
  - `${DATASET_PATH}` is the path to a directory which has a structure like below
```
├── batch_1
│   ├── batch_12_norm
│   │   ├── x.pcd
│   │   └── y.pcd
│   └── batch_12_norm_annot_seg
│       ├── x.npy
│       └── y.npy
└── batch_2
    ├── batch_20_norm
    │   ├── x.pcd
    │   └── y.pcd
    └── batch_20_norm_annot_seg
        ├── x.npy 
        └── y.npy
    ...
```
- `docekr ps -a | grep armanzarei/lettuce-soil-segmentation-train` to find the id of the container
- `docker cp ${CONTAINER_ID}:/src/model.pth ${PATH_TO_SAVE}/model.pth`
- `docker rm ${CONTAINER_ID}`

## Annotator
image: `armanzarei/lettuce-soil-segmentation-annotate`

`docker run --rm --gpus all -v ${PATH_TO_DATA}:/src/data armanzarei/lettuce-soil-segmentation-annotate`
- `${PATH_TO_DATA}` is the path to the directory which contains a bunch of `.ply` format files

:black_medium_square: If you want to use a new trained model (rather than the pretrained model inside the container) you can use `--use_given_model` flag and mount the trained model to `/src/new_trained_model/DGCNN.pth`

`docker run --rm --gpus all -v ${PATH_TO_DATA}:/src/data -v ${PATH_TO_TRAINED_MODEL}:/src/new_trained_model/DGCNN.pth armanzarei/lettuce-soil-segmentation-annotate --use_given_model`

---

To get access to the dataset (raw/annotated) : contact `armanzarei1378[at]gmail[dot]com`

---

**Model in use**: `Dynamic Graph CNN (DGCNN)`

Other tested models: 
  - `PointNet`
  - `PointNet++`
  - `RandLANet` 

more detail: [Link](https://github.com/ArmanZarei/3D_Lettuce_Soil_Segmentation)
