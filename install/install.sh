# causal-conv1d至少要求cuda11.6，需要先安装好cuda
export CPATH=/usr/local/cuda-11.7/include:$CPATH
conda create --prefix /root/autodl-tmp/pro/conda_env/OSGNet python=3.10
ln -s /root/autodl-tmp/pro/conda_env/OSGNet /root/miniconda3/envs/OSGNet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone git@github.com:OpenGVLab/VideoMamba.git
pip install -e ./causal-conv1d
pip install -e ./mamba

pip install pandas lmdb terminaltables prettytable pyyaml tensorboard torch_kmeans einops line_profiler
cd ./libs/utils
python setup.py install --user
cd ../..
pip install numpy==1.26.4