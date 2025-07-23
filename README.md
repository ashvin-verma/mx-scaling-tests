# MX Testing on Common Models
## Setup
Install docker first. Then, run
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it \
   -v "$(pwd)":/workspace nvcr.io/nvidia/pytorch:24.07-py3
```
Then, install microxcaling,
```bash
git pull https://github.com/microsoft/microxcaling
cd microxcaling
pip install -e .
```

To find your docker container,
```bash
docker ps -a 
```
this gives you a container name, which you can use to start and enter the container
```bash
systemctl start docker
docker start <container-name>
docker exec -it <container-name> bash
```
## CNN Tests
Run
```bash
python -W ignore mx_rn_mnv2.py
```
## LM Tests
Run
```bash
python -W ignore mx_bert_phi.py
```