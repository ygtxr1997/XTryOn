# [Install CUDA for Non-Root Users](https://zhuanlan.zhihu.com/p/614819188)

```shell
ln -s /Your/Path/To/cuda-xx.xx/ /Your/Path/To/cuda/
```

```shell
nano ~/.bashrc
''' append lines
export PATH=/Your/Path/To/cuda/bin:$PATH
export LD_LIBRARY_PATH=/Your/Path/To/cuda/lib64:$LD_LIBRARY_PATH
'''
```

# [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

```shell
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/Your/Path/To/cuda/

# go to the repo directory
cd third_party/grounded_sam/

# install SAM
python -m pip install -e segment_anything

# install Grounding DINO
python -m pip install -e GroundingDINO
```

# [Check Environment]()

```shell
python -m torch.utils.collect_env
```