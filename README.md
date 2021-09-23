# PhytoOracle Lettuce Soil Annotator

## Train

- `docker run -it --gpus all -v ${DATASET_PATH}:/src/data armanzarei/lettuce-soil-segmentation-train`
  - `--num_epochs` can be passed to control the number of train epochs
- `docekr ps -a | grep armanzarei/lettuce-soil-segmentation-train` to find the id of the container
- `docker cp ${CONTAINER_ID}:/src/model.pth ${PATH_TO_SAVE}/model.pth`
- `docker rm ${CONTAINER_ID}`

---

Model in use: `Dynamic Graph CNN (DGCNN)`

---

Other tested models: 
  - `PointNet`
  - `PointNet++`
  - `RandLANet` 

more detail: [Link](https://github.com/ArmanZarei/3D_Lettuce_Soil_Segmentation)
