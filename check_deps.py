
try:
    import pandas
    print("pandas ok")
    import numpy
    print("numpy ok")
    import tifffile
    print("tifffile ok")
    import imgaug
    print("imgaug ok")
    import h5py
    print("h5py ok")
    import scipy
    print("scipy ok")
    import matplotlib
    print("matplotlib ok")
    import natsort
    print("natsort ok")
    import cellpose
    print("cellpose ok")
    import skimage
    print("scikit-image ok")
    import segmentation_models_pytorch
    print("segmentation-models-pytorch ok")
    import cv2
    print("opencv-python ok")
    import PIL
    print("pillow ok")
    import yaml
    print("pyyaml ok")
    import pydantic
    print("pydantic ok")
    import tqdm
    print("tqdm ok")
except ImportError as e:
    print(f"Failed: {e}")
