# ee474_user_server

- EVA: Emotion-based Video-conferencing App
- (GPU) server for ee474 project (user emotion detection)
- Environment created by conda


## Getting Started

Let's set up the environment

### Prerequisites

- anaconda

### Setting up the environment
- Setup anaconda environment
```
$ conda env create -f environment.yml
```

- If creating the environment with environment.yml does not work, try the following
```
$ conda env create --name ee474_user_server
$ conda install -c conda-forge opencv=3.4.2
$ conda install -c anaconda flask
$ conda install -c anaconda requests
$ conda install -c conda-forge tensorflow=1.13
$ conda install -c anaconda keras=2.3.1
$ conda install -c anaconda pillow
```

### In order to start the server
- Turn on the server
```
$ cd server
$ python app.py
```

- Flask server: http://localhost:7007
