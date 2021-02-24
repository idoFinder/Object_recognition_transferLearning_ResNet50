# Object recognition using transfer-learning with ResNet50v2

This project was done as part of the course: **"Learning, representation, and Computer Vision"** by Dr Aharon bar Hillel

In this project we used a pre-trained ResNet50v2 network (on ImageNet) to detect flowers in images.
(we used Keras implementation of ResNet50v2 running on tensorflow beckend)

The code containes various architectures we evaluated on the task of flower detection and include among the rest:
- data augmentation
- Hyperparameters tunning
- Evaluation + plotting the "worst" predictions

The best configuration was ResNet50V2 + 1 Fully connected layer (with sigmoid activation) + RandomForest classifier
