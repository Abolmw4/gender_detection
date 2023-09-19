# Gender detection using face images
**The goal of our project is to detect gender based on facial images of individuals.
The trained model has been uploaded to the following link, which contains a dataset of facial images of individuals.**
>* https://drive.google.com/file/d/1e4R7VokUzDzr9GCdZXRAarUfNLc9rIp3/view?usp=sharing
>* The accuracy of this model on the test dataset was 99.85%.

>Use **python3.7** and **requirement.txt** to implement the project.


### Dataset

```
dataset/
├── train_data/
│   ├── male/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── female/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val_data/
│   ├── male/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── female/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test_data/
    ├── male/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── female/
        ├── image1.jpg
        ├── image2.jpg
        └── ...

```

### Train

>`python main.py -t /home/example.csv -v /home/example_dir -w /home/example_dir -e 15`
>
> Options
> * -t The training dataset directory in absolute form is as follows for example: home/user/gender_detection/dataset/train_data
> * -v The training dataset directory in absolute form is as follows for example: home/user/gender_detection/dataset/val_data
> * -w The location where you want to store your weights.
> * -e number of epoch for train
> * -l learning rate
> * -b batch size

### Test
You can run the 'test.py' file to evaluate the test accuracy.

If you would like the gender classification to be performed for you, please execute the 'run.py' file.

To separate the images, proceed as follows:
>`python run.py -t /home/example_dir -o /home/example_dir -m /home/example.pt`
>
> Options
> * -i images directory
> * -o The address where you want the classified images to be stored.
> * -m The absolute address of the pre-trained model.
> * -d cuda or cpu

>You can use **create_onnxmodel.ipynb** and create **onnx model** and use it for infrence.
