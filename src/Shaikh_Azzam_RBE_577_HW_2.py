"""
Azzam Shaikh
RBE 577: Homework 2
"""

# Import libraries
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

class VehicleClassificationDataset(Dataset):
    """
    Class that inherets the Dataset class and is used to load the data
    """
    def __init__(self, root_dir, transform=None):
        """
        Constructor that defines the input arguments, classes, class to index mapping, and dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}
        self.samples = []

        # Loop through the root directory and create a list that contains the image path, the index, and which folder
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(root_dir, split)
            if split != 'test':
                for class_name in self.classes:
                    class_dir = os.path.join(split_dir, class_name)
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name],split))
            else:
                for img_name in os.listdir(split_dir):
                    img_path = os.path.join(split_dir,img_name)
                    self.samples.append((img_path, -1,split))

    def __len__(self):
        """
        Overrides the Dataset length attribute specific for this class
        """
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Overrides the Dataset getitem implementation specific for this class
        """
        img_path, label, split = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, split
    
    def get_number_of_classes(self):
        """
        Custom function to return the number of classes in the dataset
        """
        return len(self.classes)
    

class EarlyStopping:
    """
    Class to create an early stopping object
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Constructor that defines the patience (number of epochs to check if the validation loss improved), and
        the minimum delta (minimum change to be considered an improvement).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss, model):
        """
        Defines the object calling implementation and what occurs when it is called
        """
        # If the best loss is empty, save the current model
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
        # If the model improves, save the new model
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f}). Saving model...')
        # If the model doesnt improve, keep track of the number of epochs since it last improved.
        # If the number of epochs is greater than the patience, stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early stopping triggered after {self.counter} epochs of no improvement.')
                self.early_stop = True

    def restore_best_weights(self, model):
        """
        Function to restore the best weights after training
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored the best model with loss of {self.best_loss}.")

class Solution():
    """
    Class that contains the main code to run the solution
    """
    def __init__(self, model_type, root_dir, writer_path):
        """
        Initialize the device, the model type, the root directory of the folder path, the Tensorboard writer,
        and the early stopping object
        """
        # Print the PyTorch version being used
        print('Using PyTorch version:', torch.__version__)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Using device:', self.device)

        self.model_type = model_type
        self.root_dir = root_dir
        self.writer = SummaryWriter(writer_path) # 'runs/10_3_classifier_50_epochs_unfreezed_4_layer'
        self.model_save_name = writer_path+'.pth'
        self.early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)


    def run_solution(self):
        """
        Main function that runs the script
        """

        # Define the transformations to be used. The transforms match that of what is used for ImageNet
        transform = transforms.Compose([
                        transforms.ToTensor(),  # Convert image to a tensor and scale pixel values to [0, 1]
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomGrayscale(),
                        transforms.RandomRotation(45),
                        transforms.Resize(256),       # Resize the shorter side to 256 pixels
                        transforms.CenterCrop(224),   # Crop the center 224x224 pixels
                        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
                    ])

        # Load the dataset        
        dataset = VehicleClassificationDataset(self.root_dir, transform)
        print('Dataset loaded')

        # Initialize the three datasets
        train_dataset = []
        val_dataset = []
        test_dataset = []

        print('Spliting datasets...')

        # Populate the three datasets 
        for img, label, split in dataset:
            if split == 'train':
                train_dataset.append((img, label))
            elif split == 'val':
                val_dataset.append((img, label))
            elif split == 'test':
                test_dataset.append((img, label))
        print(f"Training has {len(train_dataset)}. Validation has {len(val_dataset)}. Test has {len(test_dataset)}")

        # Convert the datasets to data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Visualize a batch of the data
        for images, labels in train_loader:
            self.visualize_batch(images, labels, dataset.classes)
            break  # Show only the first batch for demonstration

        # Set random seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Load a pretrained ResNet50 
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Define new classification head
        new_classification_head = torch.nn.Linear(model.fc.in_features, dataset.get_number_of_classes())

        # Replace the resnets classification head with the new one
        model.fc = new_classification_head

        # If the model is type 1 - only train classifier
        if self.model_type == 'type1':
            # Unfreeze the fully connected layer
            for name, param in model.named_parameters():
                # Only unfreeze these layers
                if 'fc' in name:  
                    param.requires_grad = True
                # Keep the rest frozen     
                else:
                    param.requires_grad = False         

        # If the model is type 2 - train the last layer and the classifier
        elif self.model_type == 'type2':
            # Unfreeze the last block (layer4) and the fully connected layer
            for name, param in model.named_parameters():
                # Only unfreeze these layers
                if 'layer4' in name or 'fc' in name:  
                    param.requires_grad = True
                # Keep the rest frozen    
                else:
                    param.requires_grad = False          
        # If the model is type 3 - train the whole network
        elif self.model_type == 'type3':
            # Unfreeze all params
            for name, param in model.named_parameters():
                param.requires_grad = True  

        pretrained_resnet = deepcopy(model)

        # Move the model to the device        
        pretrained_resnet.to(self.device)
        
        # Print a summary of the model 
        summary(pretrained_resnet,(3, 224, 224))

        # Setup loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=pretrained_resnet.parameters(), lr=1e-3,weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

        # Define number of epochs
        NUM_EPOCHS = 50
        training = self.train(device=self.device,
                            model=pretrained_resnet,
                            train_dataloader=train_loader,
                            val_dataloader=val_loader,
                            test_dataloader=test_loader,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)
        
        # Restore best model weights after training
        self.early_stopping.restore_best_weights(pretrained_resnet)
        
        # Make the model predict the classes for the different test images
        predictions = self.run_test_step(pretrained_resnet, test_loader, self.device)

        # Visualize the results of the prediction
        for images,_ in test_loader:
            labels = predictions[0:len(images)]
            self.visualize_predictions(images, labels, dataset.classes)
            break  
        
        # Save the whole model
        torch.save(pretrained_resnet,self.model_save_name)

        # Close the writer
        self.writer.close()

    @staticmethod
    def visualize_batch(images, labels, class_names):
        """
        Function to visualize a batch
        """
        plt.figure(figsize=(11, 11))
        for i in range(len(images)):
            # Denormalize the image
            image = images[i].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)  # Clip values to ensure they are within [0, 1]
            plt.subplot(8, 4, i + 1)
            plt.imshow(image)
            plt.title(class_names[labels[i]])
            plt.suptitle('Batch Visualization')
            plt.tight_layout()
            plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_predictions(images, predictions, class_names):
        """
        Function to visualize the predictions
        """
        plt.figure(figsize=(10, 5))
        for i in range(0,10):
            # Denormalize the image
            image = images[i].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)  # Clip values to ensure they are within [0, 1]
            plt.subplot(3, 4, i + 1)
            plt.imshow(image)
            plt.title(class_names[predictions[i]])
            plt.suptitle('Test Dataset Predictions for 10 Example Images')
            plt.tight_layout()
            plt.axis('off')
        plt.show()

    @staticmethod
    def train_step(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   device):
        """
        Helper function to run a training step
        :param model: An AlexNet model object to be passed
        :param dataloader: A dataloader object to be passed
        :param loss_fn: A loss function object to be passed
        :param optimizer: A optimizer object to be passed
        :param device: A device name to be passed
        :return: Return train loss and accuracy
        """
        # Put model in train mode
        model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    
    @staticmethod
    def run_val_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device):
        """
        Helper function to run a test step
        :param model: AlexNet model object to be passed
        :param dataloader: A dataloader object to be passed
        :param loss_fn: A loss function object to be passed
        :param device: A device name to be passed
        :return: Return test loss and accuracy
        """
        # Put model in eval mode
        model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    @staticmethod
    def run_test_step(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device):
        """
        Helper function to run a test step when no labels are available.
        :param model: Model object to be passed
        :param dataloader: A dataloader object to be passed
        :param device: A device name to be passed
        :return: Return model predictions
        """
        # Put model in eval mode
        model.eval()

        # Initialize a list to store predictions
        predictions = []

        # Turn on inference context manager
        with torch.no_grad():
            # Loop through DataLoader batches
            for X, label in dataloader:
                # Send data to target device
                X = X.to(device)

                # Forward pass
                test_pred_logits = model(X)

                # Get predicted labels
                test_pred_labels = test_pred_logits.argmax(dim=1)

                # Store predictions
                predictions.extend(test_pred_labels.cpu().numpy())

        return predictions
    

    def train(self,
              device,
              model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.Optimizer,
              loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
              epochs: int = 5):
        """
        Function to train the model
        :param device: A device name to be passed
        :param model: A specific AlexNet model object to be trained
        :param train_dataloader: A training dataset dataloader object to be passed
        :param val_dataloader: A validation dataset dataloader object to be passed
        :param test_dataloader: A testing dataset dataloader object to be passed
        :param optimizer: A optimizer object to be passed
        :param scheduler: An optimizer scheduler object to be passed
        :param loss_fn: A loss function object to be passed
        :param epochs: The number of epochs to be run. Default is 5
        :return: Returns the training results and a break flag if testing accuracy is greater than 95%
        """

        # Create empty results dictionary
        results = {"train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
                }
        
        print("=" * 40)
        print('Beginning training')
        print("=" * 40)
        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            print('Running epoch:', epoch)
            train_loss, train_acc = self.train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            val_loss, val_acc = self.run_val_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)

            # Send data to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Print out what's happening
            print(
                f"\nEpoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f} | "
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            # Check if we need to stop
            self.early_stopping(val_loss,model)

            if self.early_stopping.early_stop:
                print('Stopping now.')
                break

            # Step the LR scheduler
            scheduler.step(val_loss)

        # Return the filled results at the end of the training and the break flag status
        return results
    
def main():
    """
    Main function of the script
    """
    # Define the dataset location
    root_dir = 'dataset/'

    # Define the location for the tensoboard data
    writer_path = 'runs/example_folder'

    # Define the model type. Either 'type1', 'type2', or 'type3'
    model_type = 'type1'

    # Initialize a solution object
    solution = Solution(model_type, root_dir, writer_path)

    # Run the solution object
    solution.run_solution()

if __name__ == '__main__':
    main()