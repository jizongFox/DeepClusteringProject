## Information based Deep Clustering: An experimental study
___
This code accompanies the paper `Information based Deep Clustering: An experimental study`.

##### Abstract:
Recently, two methods have shown outstanding perfor-mance for clustering images and jointly learning the fea-ture representation.  The first, called Information Maximiz-ing Self-Augmented Training (IMSAT), maximizes the mu-tual information between input and clusters while using aregularization term based on virtual adversarial examples.The second, named Invariant Information Clustering (IIC),maximizes the mutual information between the clustering ofa sample and its geometrically transformed version.  Thesemethods use mutual information in distinct ways and lever-age different kinds of transformations.  This work proposesa comprehensive analysis of transformation and losses fordeep clustering, where we compare numerous combinationsof  these  two  components  and  evaluate  how  they  interactwith one another.  Results suggest that mutual informationbetween a sample and its transformed representation leadsto  state-of-the  art  performance  for  deep  clustering,  espe-cially when used jointly with geometrical and adversarialtransformations.


___
How to use:
```
git clone --recursive https://github.com/jizongFox/DeepClusteringProject.git
```
If you use `conda`, you'd prefer using a virtual environment:    
```
conda create -p env python=3.7
conda activate env
```   
Then install backend library:
```bash
cd library/deep-clustering-toolbox
python setup install
```   
The installed package support `Apex` to speed up and reduce the memory usage. Following [installation guide](https://github.com/NVIDIA/apex)
for a faster training.

The `main.py` is the main entrance of the program and the `trainer/*` defines the training behaviors. 

## to reproduce the performance of the paper `Invariant Information Clustering for Unsupervised Image Classification and Segmentation`, go to [here](https://github.com/jizongFox/DeepClusteringProject/blob/master/baseline/run_baseline.sh)
## to reproduce the performance of the paper `Learning Discrete Representations via Information Maximizing Self-Augmented Training`, go to [here](Learning Discrete Representations via Information Maximizing Self-Augmented Training)
