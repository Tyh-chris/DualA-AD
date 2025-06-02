# DualA-AD

## running: 

Modify the dataset and the dataset path of train_two.py, and it can be run.

Steps：

1. choose dataset：

   ```
   type = 'ZhangLabData'  # chexpert， VinCXR
   ```

2. imagepath

   ```
   image_path = os.path.join('../data/chest', type)
   ```

3. run train/train.py，or You can directly right-click running

   ```
   python train.py
   ```

## Evaluate

```
python eval.py
```



## datasets: 

1. ZhangLab: https://data.mendeley.com/datasets/rscbjbr9sj/3
2. Chexpert：https://github.com/tiangexiang/SQUID, it provides datasets uploaded to google drive(https://drive.google.com/file/d/14pEg9ch0fsice29O8HOjnyJ7Zg4GYNXM/view?usp=sharing
3. VinCXR: https://github.com/caiyu6666/DDAD, it provides datasets Med-AD, including divided datasets.

Place the image under the /data/chest folder, Image format:
```{
  "train": {
    "normal": ["*.png or *.png", ], # normal training images
  },
  
  "test": {
  	"normal": ["*.png or *.png", ],  # normal testing images
  	"abnormal": ["*.png or *.png", ]  # abnormal testing images
  }
}
```

## MAE model

You can use the paper (Masked Autoencoders Are Scalable Vision Learners) provided by making web site (https://github.com/facebookresearch/mae), Find the corresponding pre-trained vit-base weights. Then place the model in network/model_pth

