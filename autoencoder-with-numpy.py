from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import pickle
import torch
import torchvision
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Create solution folder
Path("solution/").mkdir(exist_ok=True)

class Layer:
    def forward(self, x):
        """Forward pass of the layer.
        For convenience, input and output are stored in the layer. 

        Args:
            x: Input for this layer. Shape is (batch_size, num_inputs).

        Returns:
            x: Output of this layer. Shape is (batch_size, num_outputs).
        """
        raise NotImplementedError

    def backward(self, gradient):
        """Backward pass of the layer.
        The incoming gradients are stored in the layer. 

        Args:
            gradient: Incoming gradient from the next layer. Shape is (batch_size, num_outputs).

        Returns:
            gradient: Gradient passed to previous layer. Shape is (batch_size, num_inputs).
        """
        raise NotImplementedError

    def update(self, learn_rate):
        """Perform weight update based on previously stored input and gradients.

        Args:
            learn_rate: Learn rate to use for the update.
        """

class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, seed=None):
        """Initialize the layer with random weights."""
        # Initialize weights with the He initializer
        rnd = np.random.RandomState(seed).randn(input_dim, output_dim)
        self.w = rnd * (2 / input_dim) 
        
        # Initialize bias with zeros
        self.b = np.zeros(output_dim,dtype=np.float64)
        
    def forward(self, x):
        """Forward pass of the layer."""
        self.input = x # Store input
       
        # ********************
        # TODO Compute output
        x = np.matmul(self.input,self.w, dtype=np.float64) +self.b

        
        # ********************
        
        self.output = x # Store output
        return x
    
    def backward(self, gradient):
        
        """Backward pass of the layer."""
        
        self.gradient = gradient # Store incoming gradient
        
        # ********************

        # TODO Apply transfer gradient
        
        #print(self.gradient.shape,"self.gradient shape in layer",self.w.T.shape,"self.w.T for layer")
        #gradient =self.gradient@self.w.T
        gradient =np.matmul(self.gradient,self.w.T, dtype=np.float64)
            

        # ********************
        
        return gradient
    
    def update(self, learn_rate):
       
        """Perform weight update"""
       
        # ********************
        # TODO Update weights and bias
        self.w -=learn_rate*np.matmul(self.input.T,self.gradient, dtype=np.float64)
      
        self.b -=learn_rate* np.mean(self.gradient.T,axis=1, keepdims=True,dtype=np.float64).reshape(-1)

        # ********************


class ReLULayer(Layer):      
    def forward(self, x):
        """Forward pass of the ReLU layer."""
        self.input = x # Store input
        
        # ********************
        # TODO Compute output
        x = np.maximum(0, self.input,dtype=np.float64)
        
#         print(self.input.shape,"relu shape forward")
        
        # ********************
        
        self.output = x # Store output
        return x
    
    def backward(self, gradient):
        """Backward pass of the ReLU layer."""
        self.gradient = gradient # Store incoming gradient
        
        # ********************
        # TODO Apply transfer gradient
        gradient[self.output<=0]=0 
#         print(self.output.shape,"relu shape backward")
        """##output or input"""
        
        # ********************
        
        return gradient

        class SoftmaxLayer(Layer):      
            def forward(self, x):
                """Forward pass of the softmax layer."""
                self.input = x # Store input
                
                # ********************
                # TODO Compute output
                # x = ...
                
                exp_score=np.exp(self.input-np.max(self.input))
                x=exp_score/np.sum(exp_score,dtype=np.float64)#,axis=1,keepdims=True)
             
                
                # ********************
                
                self.output = x # Store output
                return x
            
            def backward(self, gradient):
                """Backward pass of the softmax layer."""
                self.gradient = gradient # Store incoming gradient
                
                # ********************
                # TODO Apply transfer gradient
                exp_score=np.exp(self.gradient-np.max(self.gradient),dtype=np.float64)
                self.gradient=exp_score/np.sum(exp_score)#,axis=1,keepdims=True)
             
                
                # ********************
                
    
                # ********************
                
                return gradient


class FeedForwardNet:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        """Forwar pass through the entire network."""
        # ********************
        # TODO Compute output
        # x = ...
        for layer in self.layers:
            x=layer.forward(x)
           
        
        #print("feed forward",x.shape)
        
        # print("feed forward end")
        # ********************
        return x
    
    def backward(self, gradient):
        """Backward pass through the entire network."""
        # ********************
        # TODO Back propagate gradients through all layers
        for layer in self.layers[::-1]:
            gradient=layer.backward(gradient)
            #print("gradient.shape feednet backward ",gradient.shape)
            
        
        #print("feed backward end")
        # ********************

    def train(self, x, target, learn_rate):
        """Train one batch."""
        gradient = self.forward(x) - target  # Assumes quadratic loss function
        #print(gradient.shape,"gradient.shape in feednet train")
        self.backward(gradient)  # Backprop
        
        # Update weights in all layers
        #print("feed train start")
        for layer in self.layers:
            layer.update(learn_rate)
        #print("feed train complete")
            
        



# Load MNIST dataset
def load_mnist():
    global mnist,mnist_test,data_loader 
    mnist = torchvision.datasets.MNIST(root='data/', download=True, transform=transforms.ToTensor())
    mnist_test = np.array([x.numpy() for x, y in torchvision.datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor())],dtype=np.float64).reshape(-1, 28, 28)
    data_loader = torch.utils.data.DataLoader(mnist, batch_size=32, shuffle=True)

    # Show examples
    plt.figure(figsize=(16,2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        
        # Choose first example with the corresponding digit
        example = next(x for x, y in mnist if y == np.random.randint(10)).reshape(28, 28)#if y == i
        plt.imshow(example, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.tight_layout()

def create_autoencoder():
    global autoencoder 
    autoencoder = FeedForwardNet([LinearLayer(784, 256, seed=10),
                          ReLULayer(),
                                 LinearLayer(256, 64, seed=50),
                         ReLULayer(),
                                 LinearLayer(64, 16, seed=4),
                          ReLULayer(),
    #                               LinearLayer(128, 64, seed=0),
    #                       ReLULayer(),
    #                               LinearLayer(64, 32, seed=0),
    #                       ReLULayer(),
    #                               LinearLayer(32, 16, seed=0),
    #                       ReLULayer(),
    #                               LinearLayer(16, 32, seed=0),
    #                       ReLULayer(),
    #                               LinearLayer(32, 64, seed=0),
    #                       ReLULayer(),
    #                               LinearLayer(64, 128, seed=0),
    #                       ReLULayer(),
                                 LinearLayer(16, 64, seed=5),
                         ReLULayer(),
                                 LinearLayer(64, 256, seed=42),
                          ReLULayer(),
                                  LinearLayer(256, 784, seed=31),
                                  
                            #SoftmaxLayer(),
                         ])
    return autoencoder
def train():
    load_mnist()
    autoencoder=create_autoencoder()
    epochs = 10
    print(len(data_loader))
    losses = np.empty((epochs, 2))
    if input("enter 'd' for load model else enter")=="t":
        try:
             
            filehandler = open("./solution/a2b.pickle", 'rb') 
            autoencoder = pickle.load(filehandler)
            print("loaded")
        except:
            pass
    else:
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                running_loss = 0.0
                for batch, _ in data_loader:
                    # ********************
                    # TODO Reshape and train batch
                    #print(batch.shape)
                    batch_size=batch.shape[0]
                    batch = batch.reshape(batch_size, 28 * 28)
                    batch=batch.numpy()
                    #print(batch.shape)
                    autoencoder.train(batch,batch,0.0001)
                    
        #             for minibatch in batch:
        #                 minibatch=minibatch.reshape(1,28*28)
        #                 autoencoder.train(minibatch,minibatch,0.001)
        #                 running_loss += np.sum((autoencoder.layers[-1].output - minibatch)**2)
        #                 pass
                    
                    
                    # ********************

                    running_loss += np.sum((autoencoder.layers[-1].output - batch)**2)
                
                # Log losses and update progress bar
                train_loss = running_loss/len(mnist)
                validation_loss = np.sum(np.mean((autoencoder.forward(mnist_test.reshape(-1, 28*28))-mnist_test.reshape(-1, 28*28))**2, axis=0))
                losses[epoch] = [train_loss, validation_loss]
                pbar.set_description(f"Loss {train_loss:.02f}/{validation_loss:.02f}")

        # Save model
        with open("solution/a2b.pickle", "wb") as f:
            pickle.dump(autoencoder, f)
    

    # Visualize losses
    losses = np.array(losses)
    plt.plot(np.arange(len(losses)), losses[:,0], label="train")
    plt.plot(np.arange(len(losses)), losses[:,1], label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("solution/a2b-train.png")

def reconstruction():
    plt.figure(figsize=(12,6))
    for i in range(16):
        # Show image
        plt.subplot(4,8,2*i+1)
        plt.imshow(mnist[i][0].reshape((28,28)), cmap="gray")
        plt.axis("off")

        # Show reconstruction
        plt.subplot(4,8, 2*i+2)
        plt.imshow(autoencoder.forward(mnist[i][0].reshape(28*28)).reshape(28,28), cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("solution/a2b-rec.png")

def interpolation():

    global encoder,decoder, steps 
    # ********************
    # TODO Split the autoencoder at the bottleneck into encoder and decoder 
    encoder = FeedForwardNet(autoencoder.layers[:len(autoencoder.layers)//2])
    decoder = FeedForwardNet(autoencoder.layers[len(autoencoder.layers)//2:])



    # ********************

    #  Choose two images 
    temp1,temp2=np.random.randint(0,mnist.data.shape[0],size=2)
    image_a = mnist[temp1][0]
    image_b = mnist[temp2][0]

    # Compute their latent representations
    latent_a = encoder.forward(image_a.reshape(28*28))
    latent_b = encoder.forward(image_b.reshape(28*28))
    print(latent_a.shape)

    steps=10
    plt.figure(figsize=(16,2))
    for i, f in enumerate(np.linspace(0, 1, steps)):
        plt.subplot(1, steps, i+1)
        
        # ********************
        # TODO Interpolate between latent_a and latent_b with the mixing factor f
        latent =(1-f)*latent_a +f*latent_b



        # ********************
        
        plt.imshow(decoder.forward(latent).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.show()
    plt.tight_layout()
    plt.savefig("solution/a2c.png")


def latent_modelling():
    # Compute mean and std of latent states
    latent_space = encoder.forward(mnist_test.reshape(-1, 28*28))
    latent_space_mean = np.mean(latent_space, axis=0)
    latent_space_std = np.std(latent_space, axis=0)

    # Sample from latent distribution
    plt.figure(figsize=(16,2))
    for i, f in enumerate(np.linspace(0,1,steps)):
        plt.subplot(1, steps, i+1)
        
        # ********************
        # TODO Sample from the normal distribution with latent_space_mean and latent_space_std
        latent =  np.random.normal(latent_space_mean, latent_space_std, latent_space.shape[1])



        # ********************
        
        plt.imshow(decoder.forward(latent).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig("solution/a2d.png")

def main():
    while True:
        ip=input("1 for train,2 for interpolation, 3 for latent modelling ,x to exit")
        try:

            if ip=="1":
                train()
            elif ip=="2":
                interpolation()
            elif ip=="3":
                latent_modelling()    
            if ip=="x":
                break
        except Exception as e:
            if ip=="x":
                break
            print("something does't work, try running 1 2 3 in order    ",e)
            pass

def main_2():
    load_mnist()
    autoencoder=create_autoencoder()
    epochs = 10
    print(len(data_loader))
    losses = np.empty((epochs, 2)) 
    filehandler = open(r".\solution\a2b.pickle", "rb")
    
    autoencoder = pickle.load(filehandler)
    print("loaded")

if __name__ == "__main__":
    main()