#Exercise: MNIST classification

# The MNIST database (Modified National Institute of Standards and Technology database) is
#a large database of handwritten digits that is commonly used for training various image
#processing systems. The database is also widely used for training and testing in the field of
#machine learning. […] The MNIST database contains 60,000 training images and 10,000 testing
#images.”

import tensorflow as tf # Importing the TensorFlow library
import numpy as np
from tensorflow import keras
from keras.layers import Dense # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)

NUMBER_OF_CLASSES = 10  # The number of classes (digits 0-9 for MNIST dataset)
EPOCHS = 400  # The number of times the entire training dataset will pass through the model
BATCH_SIZE = 256  # The number of samples used in one update of the model weights
VALIDATION_SPLIT = .2  # The proportion of the training data to be used as validation data
NUMBER_OF_HIDDEN_NEURONS = 32  # The number of neurons in the hidden layer of the neural network


class MNIST_data:
    def __init__(self):
        # Load MNIST dataset
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = keras.datasets.mnist.load_data()
        (self.X_train_preprocessed, self.Y_train_preprocessed, self.X_test_preprocessed, self.Y_test_preprocessed) = self.preprocess_data()

    #1. Plotting the examples of images (plot_examples method in MNIST_data):
    def plot_examples(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - generate a tuple of 6 random indices of images within self.X_train
        #   - create a figure with 6 subplots and get all possible indices for subplots (already done)
        #   - use a for-loop to plot the 6 images

        ### Enter your code here ###
    
        random_indices = np.random.choice(self.X_train.shape[0], 6, replace=False) #generate a tuple of 6 random indices of images within self.X_train

        ### End of your code ###

        fig, axs = plt.subplots(2, 3)
        axs_idx = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
        ### Enter your code here ### use a for-loop to plot the 6 images:
        for i, (idx, (row, col)) in enumerate(zip(random_indices, axs_idx)): # Loop over the indices and the subplot indices
            axs[row, col].imshow(self.X_train[idx], cmap='gray')
            axs[row, col].set_title(f"Label: {self.Y_train[idx]}")
            axs[row, col].axis('off')  # Hide axis labels
        

        ### End of your code ###

        plt.show()
    
    #2. Preprocessing the data (preprocess_data method in MNIST_data)
    def preprocess_data(self):
        # Input:
        #   - self #already done in 1.
        # Return:
        #   - X_train_preprocessed
        #   - Y_train_preprocessed
        #   - X_test_preprocessed
        #   - Y_test_preprocessed
        
        # Function:
        #   - Reshape each image from 28x28 to 784x1 (input shape: (X, 28, 28), output shape: (X, 784))
        #   - The value of each pixel is in range [0, 255] -> Normalize those values to a range [0, 1]
        #   - Preprocess Y-values  (One-hot encoding of labels):
        #       - Neural net returns probability for each class -> output shape (1, 10) 
        #       - Y-value for each picture has to be an array consisting of zeros and one 1, not only a single integer
        #       e.g. Y-value for digit '3': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        #       - Y input shape: (X, ), output shape: (X, 10)
        
        #   - Assign the result of the operations to the returned variables
        
        ### Enter your code here ###
        # Preprocess X-values:
        # Reshape and normalize the input data
        X_train_preprocessed = self.X_train.reshape(-1, 28*28).astype('float32') / 255
        X_test_preprocessed = self.X_test.reshape(-1, 28*28).astype('float32') / 255
        
        #Reshaping- makes the data suitable for a neural network, which expects input data in a 1D format.
        #Normalization- scales the pixel values into a smaller, more manageable range, improving training efficiency and stability
        
        #Why: The MNIST dataset consists of images that are 28x28 pixels. In deep learning models (specifically fully connected neural networks), 
        #the input data must be in a one-dimensional array format. Therefore, we reshape each 28x28 image into a single vector 
        #of length 784 (since 28 * 28 = 784). This makes it easier to feed the image into the neural network.
        #How: Each 28x28 image is flattened into a 784-length vector. So instead of having a 2D array representing the image, we convert it 
        #into a 1D array, which is a suitable input format for fully connected layers (i.e., Dense layers).
        
        #In the MNIST dataset, each pixel has an integer value between 0 and 255. However, neural networks typically perform better and train 
        #more efficiently when input data is scaled or normalized to a smaller range, usually between 0 and 1. This is because
        #large values can cause issues in training (like gradient explosion or slower convergence).
        
        # Preprocess Y-values:
        # One-hot encoding of labels
        Y_train_preprocessed = keras.utils.to_categorical(self.Y_train, NUMBER_OF_CLASSES) #keras.utils.to_categorical is a utility function 
        #that converts these integer labels into one-hot encoded vectors.
        Y_test_preprocessed = keras.utils.to_categorical(self.Y_test, NUMBER_OF_CLASSES) 
        
        return X_train_preprocessed, Y_train_preprocessed, X_test_preprocessed, Y_test_preprocessed

        ### End of your code ###

class TrainSolver:
    def __init__(self):
        self.dataset = MNIST_data()
        self.dataset.plot_examples()
        self.mnist_model = self.create_net()

    #3. Creating the neural network model (create_net method in TrainSolver):
    #You need to create a sequential model with an input layer, one hidden layer,
    # and an output layer with the appropriate number of neurons.
    def create_net(self):
        # Input:
        #   - self 
        # Return:
        #   - model
        # Function:
        #   - Create a Sequential model with 3 Layers:
        #       - Input Layer: pixels from image (hint: use model.add(keras.Input()))
        #       - Hidden Layer: Type: Dense, Activation function: ReLu, number of neurons: NUMBER_OF_HIDDEN_NEURONS
        #       - Output Layer: Type: Dense, Activation function: Softmax, number of neurons: Number of classes
        #   - Compile model:
        #       - Optimizer: Adam
        #       - Loss: Mean-squared-error
        #       - Metrics: Accuracy
        #   - Print model summary and return model
        model = keras.Sequential([
            keras.Input(shape=(28*28,)),  # Input layer: Flattened 28x28 image
            Dense(NUMBER_OF_HIDDEN_NEURONS, activation='relu'),  # Hidden layer
            Dense(NUMBER_OF_CLASSES, activation='softmax')  # Output layer
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.summary()
        return model


    def train(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - train model and store result to variable history:
        #       - Use model.fit()
        #       - Specify batch_size, epochs, verbose=2 and validation_split (use global variables)
        #   - evaluate model with test data
        #       - hint: use model.evaluate()
        #   - plot loss and accuracy (already done)

        ### Enter your code here ###
        # Training the model
        history = self.mnist_model.fit(self.dataset.X_train_preprocessed, 
                                       self.dataset.Y_train_preprocessed,
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE,
                                       validation_split=VALIDATION_SPLIT,
                                       verbose=2)
        
        # Evaluate the model
        test_loss, test_acc = self.mnist_model.evaluate(self.dataset.X_test_preprocessed, 
                                                        self.dataset.Y_test_preprocessed)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        # Plot loss and accuracy
        df_loss_acc = pd.DataFrame(history.history)
        df_loss_acc.describe()


        # Extract data for loss and accuracy (below)
        #df_loss = df_loss_acc[['loss', 'val_loss']].rename(columns={'loss': 'Loss', 'val_loss': 'Validation Loss'})
        #df_acc = df_loss_acc[['accuracy', 'val_accuracy']].rename(columns={'accuracy': 'Accuracy', 'val_accuracy': 'Validation Accuracy'})


        ## End of your code ###

        # Code taken from: https://newbiettn.github.io/2021/04/07/MNIST-with-keras/
        # Convert this object to a dataframe
        df_loss_acc = pd.DataFrame(history.history)
        df_loss_acc.describe()

        # Extract to two separate data frames for loss and accuracy
        df_loss = df_loss_acc[['loss', 'val_loss']]
        df_loss = df_loss.rename(columns={'loss': 'Loss', 'val_loss': 'Validation Loss'})
        df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
        df_acc = df_acc.rename(columns={'accuracy': 'Accuracy', 'val_accuracy': 'Validation Accuracy'})

        # Plot the data frames
        df_loss.plot(title='Training vs Validation Loss', figsize=(10, 6))
        df_acc.plot(title='Training vs Validation Accuracy', figsize=(10, 6))
        plt.show()


if __name__ == "__main__":
    session = TrainSolver()
    session.train()
#313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 278us/step - accuracy: 0.9509 - loss: 0.5358
#Test Loss: 0.44905468821525574, Test Accuracy: 0.9592000246047974

#The model demonstrates high accuracy and low loss on the test data, 
#indicating good training quality and no significant overfitting.