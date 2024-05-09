This is the read me file for the project "Forest Fire Detection Using CNN".
The dataset for this model was selected on Kaggle.
The link for the dataset is as follows:
https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data


In this project we are using CNN to train a model inorder to identify forests on fire. The model is able to identify forests on fire with a Testing Accuracy of 92.74%.

Convolutional Layers:
	
Conv2D (32 filters, kernel size 3x3, ReLU activation):
	This layer applies 32 convolutional filters of size 3x3 to the input image.
	The ReLU activation function is used to introduce non-linearity into the network.

MaxPooling2D (pool size 2x2):
	Max pooling reduces the spatial dimensions of the feature maps by taking the maximum value in each 2x2 region.
	It helps in reducing the computational complexity and prevents overfitting by introducing translation invariance.
	
Conv2D (32 filters, kernel size 3x3, ReLU activation):
	Another convolutional layer with 32 filters and a 3x3 kernel size.
	The ReLU activation function is applied to introduce non-linearity.

MaxPooling2D (pool size 2x2):
	Another max pooling layer to reduce the spatial dimensions further.

Conv2D (64 filters, kernel size 3x3, ReLU activation):
	A convolutional layer with 64 filters and a 3x3 kernel size.
	ReLU activation is applied.

Dropout (rate 0.2):
	Dropout randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting.

MaxPooling2D (pool size 2x2):
	Another max pooling layer to further reduce spatial dimensions.

Conv2D (128 filters, kernel size 3x3, ReLU activation):
	A convolutional layer with 128 filters and a 3x3 kernel size.
	ReLU activation is applied.

SpatialDropout2D (rate 0.4):
	Spatial dropout randomly sets a fraction of feature map values to 0, which helps prevent overfitting of neighboring pixels.

Fully Connected Layers:

Flatten:
	Flattens the 3D output of the convolutional layers into a 1D vector for input to the fully connected layers.

Dense (256 units, ReLU activation):
	A fully connected layer with 256 units and ReLU activation.
	Introduces non-linearity to the network.

Dropout (rate 0.4):
	Dropout layer to prevent overfitting.

Dense (256 units, ReLU activation):
	Another fully connected layer with 256 units and ReLU activation.
	Helps in learning complex patterns in the data.

Dropout (rate 0.2):
	Another dropout layer to prevent overfitting.

Dense (1 unit, Sigmoid activation):
	Final output layer with 1 unit and sigmoid activation.
	Produces the probability of the input belonging to a particular class (binary classification).
