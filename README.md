## MDEformer: A Spatio-Temporal Decoupling Transformer with the Multidimensional Information Encoding for Certain Traffic Flow Prediction

#### Required Packages
```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

#### Training Commands

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
- PEMS03
- PEMS04
- PEMS07
- PEMS08
