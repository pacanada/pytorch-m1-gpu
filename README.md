# pytorch-m1-gpu
## Getting started

1. Install native python for m1 [miniconda3 MacOs m1](https://docs.conda.io/en/latest/miniconda.html)

2. Run:

```shell
/Users/<username>/miniconda3/condabin/conda create -n torch-nightly python=3.8
```
```
/Users/<username>/miniconda3/envs/torch-nightly/bin/pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
(to avoid configuring miniconda for native python as default)

3. Run the script:

```shell
/Users/<username>/miniconda3/envs/torch-nightly/bin/python3 main.py
```
Alternatively you can activate the conda env with the cexisting conda instalation
```shell
conda activate /Users/pablocanadapereira/miniconda3/envs/torch-nightly
```

4. Check python version and if mps backend is supported
```shell
python -c 'import platform; print(platform.platform())'
```
```shell
print(torch.backends.mps.is_available())
```

For now the performance is way worse..

