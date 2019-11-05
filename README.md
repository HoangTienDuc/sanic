# FTID

End-to-End Vietnamese ID card OCR system:

+ Architecture:

![architecture](https://hiepph.github.io/ocr/architecture.png)

+ Backend:

![backend](https://hiepph.github.io/ocr/Backend.png)


## Setup

### Requirements

+ Python >= 3.5, < 3.7

+ Tensorflow >= 1.9

+ Pip packages: `pip install -r requirements.txt`

+ Tensorflow Serving:

```
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server
```


### Up-and-running

#### Tensorflow Serving

+ Prepare config file:

```
cp $PWD/resources/configs/tensorflow/serving.config.example $PWD/resources/configs/tensorflow/serving.config
```

+ Replace `/share/export/{Reader,Classifier,Detector}` with the absolute path to your exported models.

+ Boot up tensorflow serving:

```
tensorflow_model_server --port=8500 --model_config_file=/absolute/path/to/your/serving.config
```


Or through nvidia-docker:

```
NV_GPU=0 nvidia-docker run -it -p 8500:8500 -v $PWD/export:/share/export -v $PWD/resources/configs/tensorflow/serving.config.example:/share/serving.config -v $PWD/resources/configs/tensorflow/batching_config.txt:/share/batching_config.txt -t tensorflow/serving:latest-devel-gpu tensorflow_model_server --model_config_file=/share/serving.config --enable_batching=true --batching_parameters_file=/share/batching_config.txt
```

### Main service

#### Development with Sanic

+ Prepare env file:

```
touch.env
```

+ Config:

```
# Important
export FTI_IDCARD_HOME=/path/to/FTI/ftid # or simple `pwd`
export PYTHONPATH=$FTI_IDCARD_HOME

# Optional
# Tf serving host
export FTI_IDCARD_SERVING=10.3.9.222:8500
# Number of workers to run
export FTI_IDCARD_WORKERS=4
# Service host
export FTI_IDCARD_HOST=0.0.0.0
# Serve port
export FTI_IDCARD_PORT=5000
```

+ Boot up service:

```
source .env
python main.py
```


## Deploy

Deploy with Docker with default exposed port 5000 for `api` and 8500 for `serving`.


+ Shallow clone this repository:

```
git clone --depth 1 https://gitlab.com/fpt_id_card/ftid
```

+ Copy `export` model (which contains `export/{Reader,Detector,Classifier}`, you can get it from `fti01:/home/ducna/share/export`) to this repository

+ Install Docker Compose:

```
pip install docker-compose
```


### CPU

![cpu](https://hiepph.github.io/ocr/cpu.png)

```
docker-compose up
```

### GPU

![gpu](https://hiepph.github.io/ocr/gpu.png)


+ Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

+ Choose GPUs to use:

```
export CUDA_VISIBLE_DEVICES='0'
```

+ Then:

```
docker-compose -f nvidia-docker-compose.yml up
```


## Benchmark

+ Install [wrk2](https://github.com/giltene/wrk2)

+ Benchmark:

    ```
    wrk -t2 -c100 -d30s -R2000 -s benchmark.lua http://localhost:5000
    ```

## Kubernetes deployment

+ [currently **deprecated**] With Rook, copy export model + config file to `rook-tools` pod:

    ```
    export ROOK_POD=rook-ceph/$(kubectl get pods -n rook-ceph -l app=rook-ceph-tools -o jsonpath="{.items[0].metadata.name}")
    kubectl cp resources/configs/tensorflow $ROOK_POD:/ceph/vision/share/config
    kubectl cp export/ $ROOK_POD:/ceph/vision/share/export
    ```

+ Deploy (CPU):

    ```
    kubectl apply -f k8s-deployment.yml
    ```
