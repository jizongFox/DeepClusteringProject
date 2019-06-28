How to use:
`git clone --recursive https://github.com/jizongFox/DeepClusteringProject.git`

prefer using a virual environment:    
`conda create -p env python=3.7`   
activate the env:   
`conda activate env`
   
go to the `library/deep-clustering-toolbox` to install the backend package by:   
`cd library/deep-clustering-toolbox`   
`pip install -e .`  
Normally you will be installed a lot of packages.   
The installed package support mix precision training to speed up and reduce the memory usage. Following [installation guide](https://github.com/NVIDIA/apex)
for a faster training.

The `main.py` is the main entrance of the program and the `ClusteringGeneralTrainer.py` defines the loss and training behaviors. 

