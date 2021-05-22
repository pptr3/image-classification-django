from django.http import HttpResponse
from django.shortcuts import render
from django.template import Context
import pandas as pd
import numpy as np
import pickle
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer
from functools import partial
import copy
from django.core.files.storage import FileSystemStorage


def home(request):
    return render(request, "form.html")

def result(request):

    cnn = pickle.load(open('model_test.pkl', 'rb'))
    classes = ['airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']

    myimg = request.FILES['avatar']
    fs = FileSystemStorage()
    myimg_path  = fs.save(myimg.name, myimg)
    img_to_display_full_path  = fs.url(myimg_path)
    # remove first char since it is a "/"
    im = img_to_display_full_path[1:]
    im = cv2.imread(im)

    var_image = torch.Tensor(im)
    var_image = torch.unsqueeze(var_image, 0).permute(0, 3, 1, 2)
    scores = cnn(var_image.to('cpu'))
    y_hat = classes[torch.argmax(scores, 1)]

    context = {'myimg_path':img_to_display_full_path, 'y_hat':y_hat}

    return render(request, "result.html", context)