# Insanely-Fast-Whisper(Dockerized)

This includes instructions for running the Dockerized version of the application. The [Docker](https://hub.docker.com/r/d0ck3rize/insanelyfastwhisper) image is hosted on Docker Hub for easy access.

![Alt text](imgs/head.png)<br>

## Prerequisites

Before getting started, ensure you have [Docker](https://www.docker.com/get-started) installed on your system.

## Usage (CLI)

- #### Pull the Docker image from Docker Hub:

  ```bash
    docker pull d0ck3rize/insanelyfastwhisper:latest
  ```

  ![Alt text](imgs/1.png)
  <br>

- Verify the presence of the Docker image

      ```bash
      docker images
      ```

  ![Alt text](imgs/2.png)
  <br>

- #### Run the Docker image in a container with insanelyfastwhisper arguments

  ```bash
   docker run -it --name <my-container> d0ck3rize/insanelyfastwhisper:latest
  ```

  ![Alt text](imgs/3.png)
  <br>

- To check running containers

```bash
 docker ps
```

![Alt text](imgs/end.png)
<br>
