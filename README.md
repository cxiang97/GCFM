# GCFM
A Graph Convolutional Fusion Model for Community Detection in Multiplex Networks

# Explaination
`gcfm.py`: includes the main, train and model.
`GNN.py`: basic GNN layer
`utils.py`: includes graph loading, and feature loading
`evaluation.py`: NMI, ARI, and Purity
`multilayer_Q.py`: multilayer modularity
`multi_data`: store node initial embedding in each layer and ground truth
`multi_graph`: store the edges in each layer

# How to use
```
python gcfm.py --name mLFR200_mu0.35
```

#Requirements

- torch==1.3.1
- numpy==1.17.2
- munkres==1.0.12
- scipy==1.3.1
- scikit_learn==0.22.1
