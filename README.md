# Deployment of Machine Learning Models
This project was a part of **Data Science Specialization** from **E&ICT ACADEMY IIT ROORKEE** in association with [**Cloudxlabs**](http://cloudxlab.com/).
This project intends to show how to deploy your machine learning models to production.
We'll train the MNIST model, save the model to the file, load the model from the file in the flask app and predict the digit for the new images. Since input images in MNIST are 28x28 greyscale images, the images used for predictions have to be processed. They should be converted to greyscale and resized to 28x28 pixels. Because of this, we won't get the accuracy in predictions but we will learn how to move our model to production (and which is the sole objective of this project).

We'll use Flask for exposing the model using the REST API for predictions. Flask is a micro web framework written in Python. It's lightweight and easy to learn.

### Follow the steps mentioned below
### 1) Clone to repository
```python
git clone https://github.com/cloudxlab/ml.git
```

### 2) Set the Python path - On CloudxLab, the default installation is python2
```python
export PATH=/usr/local/anaconda/bin/:$PATH
```

### 3) Create virtual environment
```python
cd ml/projects/deploy_mnist/
virtualenv -p python3 venv
```

### 4) Activate virtual environment
**What is virtual environment?**

A virtual environment is a Python environment such that the Python interpreter, libraries and scripts installed into it are isolated from those installed in other virtual environments, and (by default) any libraries installed in a “system” Python, i.e., one which is installed as part of your operating system.

A virtual environment is a directory tree which contains Python executable files and other files which indicate that it is a virtual environment.

The venv module provides support for creating lightweight “virtual environments” with their own site directories, optionally isolated from system site directories.
We are activating the virtual environment using the following command:
```python
source venv/bin/activate
```

### 5) Install the flask and other requirements
**What does the requirements.txt file do?**

If you have browsed any python projects on Github or elsewhere, you have probably noticed a file called requirements.txt This requirements.txt file is used for specifying what python packages are required to run the project you are looking at. Typically the requirements.txt file is located in the root directory of your project. If you open the requirements.txt file, you will see the following files:

- Flask==1.0.2
- numpy==1.16.2
- #Pillow==2.2.1
- scikit-learn==0.20.3
- pillow==6.0.0
To run this project, we need to install these packages. We have saved them in the requirements.txt file so that they can be installed in one go using the following command:
```python
pip install -r requirements.txt
```

**What are these packages in the requirements.txt file?**

We already know about numpy and scikit-learn, let’s get to know the rest of them.

Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.

pillow is the friendly PIL fork by [Alex Clark](https://github.com/python-pillow/Pillow/graphs/contributors) and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors.

### 6) Train the model
The trained model will be saved in trained_models directory

```python
mkdir -p trained_models
python train_mnist_model.py
```

### 7) Start the flask server for predictions
For the API code, see the file predictions.py under flask_app directory. Run the server on port 4041. If the port is already in use then use any of the port in the range of 4040 to 4060 as on CloudxLab only these ports are open for public access.

```python
cd flask_app
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
export FLASK_APP=predictions.py
flask run --host 0.0.0.0 --port 4041
```

### 8) Predict the digit for the new image
We will use the test images for predictions. Login to another console and run below commands.

```python
cd ml/projects/deploy_mnist/
curl -F 'file=@test-images/7.png' 127.0.0.1:4041/predict
```

**What does the curl command do?**

The curl command transfers data to or from a network server, using one of the supported protocols (HTTP, HTTPS, FTP, FTPS, SCP, SFTP, TFTP, DICT, TELNET, LDAP or FILE). Let’s explain the working of the following line:

> curl -F 'file=@test-images/7.png' 127.0.0.1:4041/predict

Here we are executing the curl command to transfer data, which in this case is the 7.png file, to the server at the address 127.0.0.1:4041/predict so that, you guessed it right, the script train_mnist_model.py can predict the image.

### The REST API will return something like below JSON object

{"digit":7}

### Public API
Our flask server is running on the CloudxLab web console. Let's say our web console is e.cloudxlab.com then the end Point URL will be http://e.cloudxlab.com:4041/predict

We can call/use this REST API by using the above mentioned End Point URL.

Replace 4041 with the port number on which your server is running.
