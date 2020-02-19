import torch
import matplotlib.pyplot as plt
import utils
import dataloaders
import torchvision
import numpy as np
from trainer import Trainer

torch.random.manual_seed(0)


class FullyConnectedModel(torch.nn.Module):     # Now with hidden layer

    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28*28
        num_hidden_nodes = 64
        # Number of classes in the MNIST dataset
        num_classes = 10

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, num_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_nodes, num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28*28)
        out = self.classifier(x)
        return out


# ### Hyperparameters & Loss function

# Hyperparameters
batch_size = 64
learning_rate = .0192
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Model definition
model = FullyConnectedModel()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)
image_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize([0.5], [0.25])   # Normalizes image to range [-2,2]
])
dataloader_train, dataloader_val = dataloaders.load_dataset(batch_size, image_transform=image_transform)


trainer = Trainer(
  model=model,
  dataloader_train=dataloader_train,
  dataloader_val=dataloader_val,
  batch_size=batch_size,
  loss_function=loss_function,
  optimizer=optimizer
)
train_loss_dict, val_loss_dict = trainer.train(num_epochs)

####### CODE FOR CREATING IMAGES FOR WEIGHTS#########
#weight = next(model.classifier.children()).weight.data
#for w in range(weight.shape[0]):
#    weight_min = weight[w, :].min()
#    weight_max = weight[w, :].max()
#    
#    im = np.zeros((28, 28))
#
#    for i in range(28):
#        for j in range(28):
#            im[i, j] = float((weight[w, i * 28 + j] - weight_min)/(weight_max-weight_min))
#    plt.imsave("images/digit_" + str(w) + "_weights_illustrated.jpg", im, cmap="gray")


# Plot loss
utils.plot_loss(train_loss_dict, label="Train Loss")
utils.plot_loss(val_loss_dict, label="Test Loss")
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Number of Images Seen")
plt.ylabel("Cross Entropy Loss")
plt.savefig("training_loss.png")

plt.show()
torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_val, model, loss_function)
print(f"Final Test Cross Entropy Loss: {final_loss}. Final Test accuracy: {final_acc}")
