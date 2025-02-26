# Textile Classification

### This project focuses on classifying three types of silk thread defects:
 Sticky ,
 Silk Scraps , 
 Hairiness 

## Step 1: Preprocessing

Traditional image processing techniques (OpenCV) are used to detect defect areas in the images.

## Step 2: Image Enhancement

Since defect details are often unclear, AI Super Resolution is applied to enhance feature visibility and improve defect detection.
## Step 3: Model Training

The processed images are used to train a classification model that distinguishes between the three defect types.
## Step 4: Model Visualization

GradCAM and t-SNE are utilized to visualize and analyze the model’s classification performance.
## Summary

A comparison of different models and the impact of Super Resolution was conducted. The application of Super Resolution improved classification accuracy from 79% to 85%, demonstrating its effectiveness in enhancing defect recognition.

