# Welcome to OpenInnovI2Backend 👋
![Version](https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000)
[![Twitter: YannDurand11](https://img.shields.io/twitter/follow/YannDurand11.svg?style=social)](https://twitter.com/YannDurand11)

> Open Innovation's project IA prediction to predict actors by using his voice and training our model

## All links

* [BackOffice](https://github.com/AlbanGuillet/OpenInnovBackOffice)
* [Homepage](https://g72ze0duasao.umso.co/)
* [API](https://github.com/EddyCheval/AsaeyOinnovApi)
* [FRONT](https://github.com/SimonHuet/who-s-that-actor-front)

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

## Using Bucket S3 with aws CLI

Tutorial link : https://www.scaleway.com/en/docs/object-storage-with-aws-cli/

If you're using a S3 bucket, you need to set-up an aws config (or whatever else). This is done by editing the config file. Basically, the aws cli is installed in the requirements.txt. 

In order to edit the config file, you need to
- create the file
```sh
aws configure set plugins.endpoint awscli_plugin_endpoint
```

- access where file is
```sh
cd ~/.aws/
vim config
```

- Edit it like this : 
```
[plugins]
endpoint = awscli_plugin_endpoint

[default]
region = nl-ams
s3 =
  endpoint_url = https://s3.nl-ams.scw.cloud
  signature_version = s3v4
  max_concurrent_requests = 100
  max_queue_size = 1000
  multipart_threshold = 50MB
  # Edit the multipart_chunksize value according to the file sizes that you want to upload. The present configuration allows to upload files up to 10 GB (100 requests * 10MB). For example setting it to 5GB allows you to upload files up to 5TB.
  multipart_chunksize = 10MB
s3api =
  endpoint_url = https://s3.nl-ams.scw.cloud 
```

Precise your own region and url endpoint of yours

After this, precise your credentials in the credentials files

- create the file
```sh
aws configure
```

- Edit like this
```
[default]
aws_access_key_id=<ACCESS_KEY>
aws_secret_access_key=<SECRET_KEY>
```

## Using S3 Scaleway with s3fs

https://www.scaleway.com/en/docs/object-storage-with-s3fs/ 

Install s3fs
```sh
apt -y install automake autotools-dev fuse g++ git libcurl4-gnutls-dev libfuse-dev libssl-dev libxml2-dev make pkg-config
```

## Author

👤 **Yann Durand**

* Website: https://codewithnefaden.wordpress.com/
* Twitter: [@YannDurand11](https://twitter.com/YannDurand11)
* Github: [@Nefaden](https://github.com/Nefaden)

## Other contributors

👤 **Eddy Cheval**

* Github: [@EddyCheval](https://github.com/EddyCheval)

👤 **Alban Guillet**

* Github: [@AlbanGuillet](https://github.com/AlbanGuillet)

👤 **Simon Huet**

* Github: [@SimonHuet](https://github.com/SimonHuet)

👤 **Alexandre Rabreau**

* Github: [@AlexandreRab](https://github.com/AlexandreRab)

## Show your support

Give a ⭐️ if this project helped you!


***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
