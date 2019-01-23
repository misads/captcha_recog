# capcha_recog
基于CNN的通用型字符验证码识别  
综合准确率高于90%

### Model graph
 
![graph](http://www.xyu.ink/wp-content/uploads/2019/01/graph_run.png)

### Install
```pip install tensorflow```

　　If you want to run with GPU,you may try```pip install tensorflow-gpu```

```pip install opencv-python```

### Train
1. Prepare your dataset (png or jpg format) in './train' directory.  
　　We also offer a dataset if you do not have one, run ```unzip train.zip``` to decompress it.

2. Modify config.py to your liking, set basic parameters like IMAGE_HEIGHT, IMAGE_WIDTH, CAPTCHA_LENGTH and CHAR_SET_LEN.

3. run
```python train.py```  
　　After training the steps you set (in config.py) the model will be saved automatically.

### Test
1. Prepare your test dataset or run ```unzip test.zip``` to use our's.

2. run ```python test.py``` to load model and predict all capchas in './test'  
you can also modify test.py to test only one image.

### Blog

[capcha recognition based on CNN](http://www.xyu.ink/425.html "教务处验证码识别")