# Watermark Removal Convolutional Neural Network
Machine Learning Engineer (aspiring): Ian H. Ryan<br>
Version: 0.1<br>
Timeline: May 22, 2025 - June 24th, 2025<br><br>

Upon making a variety of image scrapers for the world wide web, I noticed that a portion of the images I found contained watermarks. This sprung the idea of creating a deep learning model with the use case of removing watermarks from images. I found a research paper on ArXiv called <a href="https://arxiv.org/html/2403.05807v1" target="_blank">
        A self-supervised CNN for image watermark removal
    </a>. I used it as a reference for this project.

# Environment
- **Make & Model**:        Alienware m15 R7
- **GPU**:                 NVIDIA GeForce RTX 3060 Mobile (6 GB VRAM)
- **Secondary GPU**:       Integrated AMD Radeon Graphics
- **CPU**:                 AMD Ryzen 7 6800H (16 threads @ 4.78 GHz)
- **RAM**:                 16 GB DDR5
- **CUDA Version**:        CUDA Version: 12.8
- **Operating System**:    Pop!_OS 22.04 LTS
- **Kernel**:              6.12.10-76061203-generic

# Notebook Index
## **Table of Contents**
- [Libraries & Imports](#libraries--imports)
  - [Libraries](#libraries)
  - [Imports](#imports)
- [CUDA](#icuda)
  - [Check GPU Availability for CUDA](#check-gpu-availability-for-cuda)
  - [CUDA allocation limiting ~64mb](#cuda-allocation-limiting-64mb)
- [Seed](#seed)
- [Dataset Description](#datasetinfo)
- [CNN Model](#cnn-model)
  - [Model Architecture](#model-arch)
  - [MixedLoss Loss Function](#loss-func)
  - [Total Parameters](#tot-param)
- [Data Generation (Self-Supervised)](#data-gen-ss)
  - [DataSet Class](#data-class)
- [Hyperparameters](#hyperparameters)
  - [Batch Size, Epochs, Learning Rate, Weight Decay](#batch-epoch-lr-weightdecay-steps)
- [Transform](#transform)
- [Pathing](#pathing)
- [Data Loader](#data-loaders)
- [Criterion, Optimizer, & Scaler for AMP](#crit-opt-scaler-amp)
- [Training CNN Model](#training-cnn-model)
- [Save Summary for Regression Task Report JSON](#save-sum-reg-report)
- [Log Training History](#log-training-history)
- [Create ChangeLog](#create-changelog)
- [Create ChangeLog & Config](#create-changelog--config)
- [Save Model Weights or Save Model of Training](#save-model-weights-or-save-model-of-training)
- [Test Accuracy of CNN Training](#test-accuracy-of-cnn-training)
- [Visualizations & Metrics](#vis-metrics)
  - [Metrics](#metrics)
    - [Summary](#summary)
    - [Evaluation Cell (For Restoration CNN)](#evaluate-cell)
    - [Test Results CSV & JSON](#test-results)
    - [Check class representation in the dataset](#check-class-representation-in-the-dataset)
    - [Print out some of the test images the model predicted as class 0](#print-out-some-of-the-test-images-the-model-predicted-as-class-0)
  - [Visuals](#visuals)
    - [Learning Rate Visual](#learning-rate-visual)
    - [F1-Score over Epochs Validation](#f1-score-over-epochs-validation)
    - [Training/Validation Curves](#trainingvalidation-curves)
    - [Training & Validation Loss Plots](#training--validation-loss-plots)
    - [Confusion Matrices](#confusion-matrices)
    - [Feature Maps](#feature-maps)
    - [Kernel/Visualizations](#kernelvisualizations)
    - [Gradient Visualization](#gradient-visualization)
    - [CAM/Grad-CAM](#camgrad-cam)
    - [Explainability Tools](#explainability-tools)
    - [Side-by-Side: Original, IG, and Grad-CAM](#side-by-side-original-ig-and-grad-cam)
    - [Compare with Target Class (Grad-CAM)](#compare-with-target-class-grad-cam)
- [Literature Cited](#literature-cited)
- [Environment](#environment)
- [Recommended Resources](#recommended-resources)
- [Permission](#permission)

# Dataset Description
Utilized self-supervised pairing & watermark synthesis. Thus not being reliant on finding a myriad of paired images with watermark and without watermark. <br><br>
This project is trained from data that I scraped off of the internet from various sources while making image scrapers. With that being said, I will not be providing downloads to the data.<br>
- 1000 images, 996 jpg, 4 png file types.<br>
<br> 
Aim to create or find a dataset that is diverse in textures, colors, brightness, edges, and backgrounds.

<p>
    If you need datasets consider checking out: 
    <a href="https://www.kaggle.com/datasets/" target="_blank">
        Kaggle | 
    </a>
    <a href="https://public.roboflow.com/" target="_blank">
        Roboflow | 
    </a>
    <a href="https://cocodataset.org/#download" target="_blank">
        COCO
    </a>
</p>

# Heterogeneous U-Net CNN Architecture


# Referenced Paper
https://arxiv.org/html/2403.05807v1
<a href="https://arxiv.org/html/2403.05807v1" target="_blank">
        A self-supervised CNN for image watermark removal
    </a> by Chunwei Tian, Member, IEEE, Menghua Zheng, Tiancai Jiao, Wangmeng Zuo, Senior Member, IEEE, Yanning Zhang, Senior Member, IEEE, Chia-Wen Lin, Fellow, IEEE from 9 Mar 2024
