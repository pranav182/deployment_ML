# Deployment of Machine Learning Models
This project intends to show how to deploy your machine learning models to production.
We'll train the MNIST model, save the model to the file, load the model from the file in the flask app and predict the digit for the new images. Since input images in MNIST are 28x28 greyscale images, the images used for predictions have to be processed. They should be converted to greyscale and resized to 28x28 pixels. Because of this, we won't get the accuracy in predictions but we will learn how to move our model to production (and which is the sole objective of this project).

We'll use Flask for exposing the model using the REST API for predictions. Flask is a micro web framework written in Python. It's lightweight and easy to learn.
