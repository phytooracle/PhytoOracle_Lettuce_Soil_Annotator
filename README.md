# PhytoOracle Lettuce Soil Annotator

## Train

- `docker run -it --gpus all -v ${DATASET_PATH}:/src/data armanzarei/lettuce-soil-segmentation-train`
  - `--num_epochs` can be passed to control the number of train epochs
- `docekr ps -a | grep armanzarei/lettuce-soil-segmentation-train` to find the id of the container
- `docker cp ${CONTAINER_ID}:/src/model.pth ${PATH_TO_SAVE}/model.pth`
- `docker rm ${CONTAINER_ID}`

## Annotator

`docker run --rm --gpus all -v ${PATH_TO_DATA}:/src/data armanzarei/lettuce-soil-segmentation-annotate`
- `${PATH_TO_DATA}` is the path to the directory which contains a bunch of `.ply` format files

✅ If you want to use a new trained model (rather than the pretrained model inside the container) you can use `--use_given_model` flag and mount the trained model to `/src/new_trained_model/DGCNN.pth`

`docker run --rm --gpus all -v /home/arman/Projects/PhytoOracle_Lettuce_Soil_Annotator/data:/src/data -v /home/arman/Projects/PhytoOracle_Lettuce_Soil_Annotator/annotate/pretrained_model/DGCNN.pth:/src/new_trained_model/DGCNN.pth armanzarei/lettuce-soil-segmentation-annotate --use_given_model`


---

Model in use: `Dynamic Graph CNN (DGCNN)`

---

Other tested models: 
  - `PointNet`
  - `PointNet++`
  - `RandLANet` 

more detail: [Link](https://github.com/ArmanZarei/3D_Lettuce_Soil_Segmentation)
