# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3:Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.
## PROGRAM

### Name: Tarun S
### Register Number: 212223040226
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x



```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
optimiser = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

```python
# Train the Model
import torch

def train_model(model, train_loader, num_epochs=3, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to("cuda" if torch.cuda.is_available() else "cpu"), labels.to("cuda" if torch.cuda.is_available() else "cpu")

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: Kesava Sai')
        print('Register Number: 212223230105')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch


![image](https://github.com/user-attachments/assets/0baafae9-c8ab-4b70-85f5-38efb87d91cb)



### Confusion Matrix


![image](https://github.com/user-attachments/assets/04a477e2-1352-44b0-8fd5-78c05fd421ab)





### Classification Report

![image](https://github.com/user-attachments/assets/efc99bbe-e6a5-4060-9a07-1c73a521a995)






### New Sample Data Prediction


![image](https://github.com/user-attachments/assets/bb2729f9-ae65-4c94-92bf-c12f2328bc12)





## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
