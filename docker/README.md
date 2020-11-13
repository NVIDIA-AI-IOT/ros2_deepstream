# Run ROS2-DeepStream in Docker

For more Jetson dockers, please look at [jetson-containers](https://github.com/dusty-nv/jetson-containers) github repository.

## Docker Default Runtime

To enable access to the CUDA compiler (nvcc) during `docker build` operations, add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

You will then want to restart the Docker service or reboot your system before proceeding.

## Building the Containers

Run the following commands to build the dockerfile:

`cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .`

``` sh docker_build.sh ``` <br/>
Once you sucessfully build, you will have a ros2-eloquent container with all necessary packages required for this repository.<br/>


## Run Container

``` sh docker_run.sh ```<br/>
This will initialize docker. Clone this repository using following command and follow build and run instructions for ros2 package from here.<br/>



