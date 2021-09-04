# transformer-ocr
ocr with vision transformer

## Overview
Simple and understandable vision transformer sytle ocr project.  
The model in this repository heavily relied on high-level open source projects like [timm](https://github.com/rwightman/pytorch-image-models) and [x_transformers](https://github.com/lucidrains/x-transformers).  
And also you can find that the procedure of training is intuitive thanks to legibility of [pytorch-lightning](https://www.pytorchlightning.ai/).

## Performance  
_With private korean handwritten text dataset, the accuracy(exact match) is 95.6%._

## Data
```
./data/
├─ preprocessed_image/
│  ├─ cropped_image_0.jpg
│  ├─ cropped_image_1.jpg
│  ├─ ...
├─ train.txt
└─ val.txt
```

```text
# train.txt
cropped_image_0.jpg\tHello World.
cropped_image_1.jpg\tvision-transformer-ocr
...
```

You should preprocess the data first. Crop the image by word or sentence level area. Put all image data in specific directory. Ground truth information would be provided with `txt` file. In the file, write `\t` seperated `image file name` and `label` in same line.

## Configuration
In `settings/` you can find `default.yaml`. You can set almost every hyper-parameters in that file. Copy and change it with your experiment version. I recommend you to run with the default setting first, before you change it.

## Train  
```bash  
python run.py --setting settings/default.yaml --version 0 --max_epochs 100 --num_workers 16 --batch_size 128
```

You can check your training log with `tensorboard`.  
```bash
tensorboard --log_dir tb_logs --bind_all
```

## Predict
It's not really hard to add prediction function to the pytorch-lightning module with fully-trained model. I will leave it empty for now. But I would glady do it if there's any request. 

enjoy the code.