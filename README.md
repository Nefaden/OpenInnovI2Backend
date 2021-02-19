# Welcome to OpenInnovI2Backend 👋
![Version](https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000)
[![Twitter: YannDurand11](https://img.shields.io/twitter/follow/YannDurand11.svg?style=social)](https://twitter.com/YannDurand11)

> Open Innovation's project IA prediction to predict actors by using his voice

## Install all dependencies

```sh
pip3 install -requirements.txt
```

## Install pulsar client

```sh
!pip3 install pulsar-client
```

## Install tensorflow
* Link : https://www.tensorflow.org/install/pip

Create a python virtual environment and a repository ./venv :
```sh
python3 -m venv --system-site-packages ./venv
```

Activate this virutal env :
```sh
source ./venv/bin/activate  # sh, bash, or zsh

. ./venv/bin/activate.fish  # fish

source ./venv/bin/activate.csh  # csh or tcsh
```

When the venv is active, the CLI prefix shall be "venv"

Install packages and dependancies in the virtual env, without modifying host system's configuration :
```sh
pip install --upgrade pip

pip list  # show packages installed within the virtual environment
```

In order to exit the venv :
```sh
deactivate  # don't exit until you're done using TensorFlow
```

Install tensorflow :
```sh
pip3 install tensorflow_io
```

## Using Docker

Adapt the dockerfile.dist file as your own dockerfile

Build dockerfile :
```sh
docker build --rm -f Dockerfile -t your_container_docker_name .
```

Executing docker container interactively :
```sh
docker exec -it [container_name] bash
``` 

And then, execute pyscript like on an usual computer

In order to exit the container : 
```sh
exit
```

If you'd modified the script or any files and don't want to rebuild, because of the long processing, do :
```sh
docker cp [path_to_your_file] [container_id]:/[path_to_cp_your_file]
```

**WARNING** Soundfile dependancie will not install libsndfile, due to linux system. So do before executing pyscript:
```sh
apt-get install libsndfile1
```

And press Y when asking
## Author

👤 **Yann Durand**

* Website: https://codewithnefaden.wordpress.com/
* Twitter: [@YannDurand11](https://twitter.com/YannDurand11)
* Github: [@Nefaden](https://github.com/Nefaden)

## Show your support

Give a ⭐️ if this project helped you!


***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_