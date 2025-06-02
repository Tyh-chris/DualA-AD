# DualA-AD

## running: 

Modify the dataset and the dataset path of train_two.py, and it can be run.

步骤：

1. 选择胸片数据集：

   ```
   type = 'ZhangLabData'  # chexpert， VinCXR
   ```

2. imagepath

   ```
   image_path = os.path.join('../data/chest', type)
   ```

3. 运行train/train.py，可直接右击running

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

将图像放于../data/chest文件夹下，图像格式
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
