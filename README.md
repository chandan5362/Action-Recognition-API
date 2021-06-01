# Human-Action-Recognition
Human action recognition is a standard Computer Vision problem and has been well studied. The fundamental goal is to analyze a video to identify the actions taking place in the video. Essentially a video has a spatial aspect to it ie. the individual frames and a temporal aspect ie. the ordering of the frames. Some actions (eg. standing, running, playing, skateboarding etc.) can probably be identified by using just a single frame but for more complex actions(eg. walking vs running, bending vs falling) might require more than 1 frame’s information to identify it correctly. Local temporal information plays an important role in differentiating between such actions. Moreover, for some use cases, local temporal information isn’t sufficient and you might need long duration temporal information to correctly identify the action or classify the video.

## Dataset
Kinetics dataset was first introduced in 2017 primarily for human action classification.t was developed by the researchers: Will Kay, Joao Carreira, Chloe Hillier and Andrew Zisserman at Deepmind. The dataset contains 400 human activity classes, within any event 400 video cuts for each activity. It has 306,245 recordings and is separated into three parts, one for preparing to have 250–1000 recordings for each class, one for approval with 50 recordings per class and one for testing with 100 recordings for every class. Each clip endures around 10s. </br>
Kinetics dataset are taken from Youtube recordings. The activities are human focussed and cover a wide scope of classes including human-object communications, for example mowing lawn, washing dishes, humans Actions e.g. drawing, drinking, laughing, pumping fist; human-human interactions, e.g. hugging, kissing, shaking hands. Since the dataset is huge and downloading each clip would be a waste of time given that we already have pre-trained models by the original author. It would be smarter to work on the pre-trained model than to train and tune it separately.
For more information on the dataset, you can follow through this [link](https://arxiv.org/abs/1705.06950)

## Model
I have used ResNet_34 3D model for human action recognition. The difference between the normal resnet model and Resnet3D model is that it uses 3d convolution layers instead of 2D convolution layer. More details about the 3D ResNe_34 model can be found [here](https://github.com/kenshohara/3D-ResNets-PyTorch). For more information about the training process, please go through this [link](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html). I have used opencv and pytorch to make the inference.

# Installation Guide
First of all, clone the repository using git. You can use the below command. You need `python3` to use this `api`.

firstly, install the Python `vivrtualenv`. to install it, use the below command.

`python -m pip install virtualenv`

Now, clone the git repository.

`git clone https://github.com/chandan5362/Human-Action-Recognition.git`

You need to install the dependencies to use the `api`. To install the dependencies please follow through.

create a python environment using `virtualenv`

`virtualenv name_of_your_env`

install the dependencies from the `requirements.txt`. Go to the root directory and type the following command.

`pip install -r requirements.txt`

You also nedd to install the `OpenCV` depending upon your system. To install it using `pip`, type the following command in your terminal.

`pip install opencv-python`

Once you are done installing the dependencies, You need to start the django server to communicate to the `Human Action Recognition` api. 
To start the server, you can type in the below command.

`python manage.py runserver`

Once the server is up and runnig, now we need to create a `POST` request to the server with the following payloads.

* `path` - Path to the existing video on your machine for the action recognition task. It is completely optional.</br>
* `link` - link to the Youtube video, on whicih you want to perform action recognition. It is also optional.

To create `POST` request, please run the `test_action.py` python script. Use the following command to run the script.
The `test_action.py` support command line argument. You can use the following command to perform action individually.

If you want to perform the recognition task on the vidoe stored on your system, type the following command in the terminal.

`python test_action.py -p path_to_your_video`

If you want to perform the action recognition on any `YouTube` video, type in the following command in the terminal.

`python test_action.py -l link_to_the_youtube_video`

Note that all the above mentioned command line arguments are completely optional. If you are in hurry, and you want to check whether this code is working or not, then I am here to make your life more easier.  I have already included a sample video in case you want to go lazy. Just run the `test_action.py` using ususal python command.

`python test_action.py`


# TODO:
* As of now, the `OpenCV` uses single core for frame extraction and decoding. So, our api only extracts the `20s` video to perform action recognition. The server might crash if you pass the video longer than `40s`.
* We can make our code more efficient using multiprocessing with `OpenCV`. I will update the code in few days. Untill then, if you want to make it more efficient, you are most welcome.
* Also, there is limitation on `fps` also. As of now, we have constrained the `fps` for video processing only upto `16fps`. So, this is good idea to start with. 
