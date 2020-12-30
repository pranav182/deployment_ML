# Deployment of Machine Learning Models
This project was a part of **Data Science Specialization** from **E&ICT ACADEMY IIT ROORKEE** in association with **Cloudxlabs**.
This project intends to show how to deploy your machine learning models to production.
We'll train the MNIST model, save the model to the file, load the model from the file in the flask app and predict the digit for the new images. Since input images in MNIST are 28x28 greyscale images, the images used for predictions have to be processed. They should be converted to greyscale and resized to 28x28 pixels. Because of this, we won't get the accuracy in predictions but we will learn how to move our model to production (and which is the sole objective of this project).

We'll use Flask for exposing the model using the REST API for predictions. Flask is a micro web framework written in Python. It's lightweight and easy to learn.

### Follow the steps mentioned below
### 1 Clone to repository
> git clone https://github.com/cloudxlab/ml.git

### 2 Set the Python path - On CloudxLab, the default installation is python2
> export PATH=/usr/local/anaconda/bin/:$PATH

### 3 Create virtual environment
> cd ml/projects/deploy_mnist/
> virtualenv -p python3 venv

### 4 Activate virtual environment
> source venv/bin/activate

### 5 Install the flask and other requirements
> pip install -r requirements.txt

### 6 Train the model
The trained model will be saved in trained_models directory

> mkdir -p trained_models
> python train_mnist_model.py

### 7 Start the flask server for predictions
For the API code, see the file predictions.py under flask_app directory. Run the server on port 4041. If the port is already in use then use any of the port in the range of 4040 to 4060 as on CloudxLab only these ports are open for public access.

> cd flask_app
> export LC_ALL=en_US.utf-8
> export LANG=en_US.utf-8
> export FLASK_APP=predictions.py
> flask run --host 0.0.0.0 --port 4041

### 8 Predict the digit for the new image
We will use the test images for predictions. Login to another console and run below commands.

> cd ml/projects/deploy_mnist/
> curl -F 'file=@test-images/7.png' 127.0.0.1:4041/predict

### The REST API will return something like below JSON object

{"digit":7}
