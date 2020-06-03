This python code implements locally adaptive activation functions 
          
root
 |
 |--main.py                             main script to be called                 
 |                                                                               
 |--models                              folder for neural network models               
 |           |--models.py                   contains definition of LeNet                                    
 |           |--preact_resnet.py            contains definition of PreActResNet18                                                                                       
 |                                                                               
 |--dataset                             empty folder in which datasets will be downloaded
 |                                          (Semeion index file is provided) 
 |--results                             empty folder in which results will be stored


Use following command at root folder

    python main.py --dataset=X1 --data-aug=X2 --model=X3 --method=X4 --n=X5

to run the code with 
        dataset=X1 
        data-augmentation=X2 
        model=X3 
        method=X4 
        n=X5
        
where 
        X1 = mnist, cifar10, cifar100, svhn, fashionmnist, kmnist, or semeion
        X2 = 0 or 1
        X3 = LeNet or PreActResNet18
        X4 = 0, 1, 2, or 3
        X5 = float (default X5=1.0)
        
X1 = YY             -> experiment with YY dataset
        
X2 = 0              -> experiment without data augmentation        
X2 = 1              -> experiment with data augmentation        

X3 = LeNet          -> experiment with the LeNet model
X3 = PreActResNet18 -> experiment with the PrectResNet18 model 
                       
X4 = 0              -> experiment with no adaptive activation
X4 = 1              -> experiment with GAAF
X4 = 2              -> experiment with L-LAAF
X4 = 3              -> experiment with N-LAAF

X5 = YY             -> experiment with n = YY

The code outputs the text format results in 
    results/X1_X2_relu_X3_X4_1.txt      (if X4 = 0)
    results/X1_X2_relu_X3_X4_X5_1.txt   (if X4 is not 0)

The code outputs the numpy array results in     
    results/X1_X2_relu_X3_X4_1.npy      (if X4 = 0)
    results/X1_X2_relu_X3_X4_X5_1.npy   (if X4 is not 0)

Let Z be the saved numpy arracy. That is,
    Z = np.load('results/X1_X2_relu_X3_X4_1.npy'), or
    z = np.load('results/X1_X2_relu_X3_X4_X5_1.npy')
    
Then, Z has the size of (1, 3, 3) and
    Z[Y1, 0, 0] = training loss at epoch Y1  
    Z[Y1, 0, 1] = training accuracy at epoch Y1     
    Z[Y1, 1, 0] = test loss at epoch Y1       
    Z[Y1, 1, 1] = test accuracy at epoch Y1      
    
In this code, PreActResNet18 is used only for cifar10, cifar100, and svhn.
This code has been tested with: Python 3.6.7 and torch 1.0.1
       