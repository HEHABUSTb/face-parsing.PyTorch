import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.spatial.distance import cosine

# === InsightFace ===
from insightface.app import FaceAnalysis

# === Face Parsing ===
import torch
from torchvision import transforms
from model import BiSeNet  # из face-parsing.PyTorch