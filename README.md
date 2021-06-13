# Locally-Adaptive-Activation-Functions-Neural-Networks-
These are the set of python codes for Locally Adaptive Activation Function (LAAF) used in deep neural networks.  

Abstract:
We propose two approaches of locally adaptive activation functions namely, layer-wise and neuron-wise locally adaptive activation functions, which improve the performance of deep and physics-informed neural networks. The local adaptation of activation function is achieved by introducing a scalable parameter in each layer (layer-wise) and for every neuron (neuron-wise) separately, and then optimizing it using a variant of stochastic gradient descent algorithm. In order to further increase the training speed, an activation slope-based slope recovery term is added in the loss function, which further accelerates convergence, thereby reducing the training cost. On the theoretical side, we prove that in the proposed method, the gradient descent algorithms are not attracted to sub-optimal critical points or local minima under practical conditions on the initialization and learning rate, and that the gradient dynamics of the proposed method is not achievable by base methods with any (adaptive) learning rates. We further show that the adaptive activation methods accelerate the convergence by implicitly multiplying conditioning matrices to the gradient of the base method without any explicit computation of the conditioning matrix and the matrix–vector product. The different adaptive activation functions are shown to induce different implicit conditioning matrices. Furthermore, the proposed methods with the slope recovery are shown to accelerate the training process.


References for adaptive activation functions:

1. A.D.Jagtap, K.Kawaguchi, G.E.Karniadakis, Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 20200334, 2020. (http://dx.doi.org/10.1098/rspa.2020.0334).

2. A.D. Jagtap, K.Kawaguchi, G.E.Karniadakis, Adaptive activation functions accelerate convergence in deep and physics-informed neural networks, Journal of Computational Physics, 404 (2020) 109136. (https://doi.org/10.1016/j.jcp.2019.109136)

3. A.D. Jagtap, Y. Shin, K. Kawaguchi, G.E. Karniadakis, Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions,
arXiv preprint, arXiv:2105.09513, 2021. (https://arxiv.org/pdf/2105.09513.pdf)
