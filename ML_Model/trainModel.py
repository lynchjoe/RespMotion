import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from Model import UNet
from customDataset import TrainTestDataset

# Modify the sample and mask directories with the data to be trained on
sample_dir = 'PATH TO THE SAMPLES'
mask_dir = 'PATH TO THE LABELS'

# These are the transformations to be applied to the training data. This inlcudes converting the sample and mask data to pytorch images, resizing them to the model's expected input size (512x512), randomly rotating the images to improve model performance, randomly flipping them to improve model performance, and normalizing them to increase the liklihood of convergance
train_transformations = v2.Compose(
        [v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=[512,512]),
        v2.RandomRotation(degrees=30),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.Normalize(mean=[0.0], std=[1.0])]
)

# Initialize a custom dataset that can be used with the pytorch DataLoader
dataset = TrainTestDataset(mask_dir=mask_dir, sample_dir=sample_dir, transform=train_transformations)

# Define train_loader
batch_size = 5
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Assign the model to the gpu if it is available. If not, assign it to the CPU. Note that using the cpu will significantly increase training time!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Define optimizer, loss, and scheduler functions
learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

# Training loop    
num_epochs = 50

for epoch in range(num_epochs):

    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device)
        label = label.to(device)
        
        # Forward Pass
        output = model(sample)
        
        # Calculate the loss and append it to the list of losses for the scheduler
        loss = criterion(output, label)

        # Backpropagation
        loss.backward()
        
        # Adjust weights and clear the gradients
        optimizer.step()
        optimizer.zero_grad()
        
    # Step the scheduler to adjust the learning rate 
    scheduler.step()
    
    print(f'Learning Rate: {scheduler.get_last_lr()[0]} per the scheduler and {optimizer.param_groups[0]['lr']} per the optimizer')
    print(f'Epoch: {epoch+1}/{num_epochs}, loss: {loss.item():.4f}')

# Save the model. Replace with model name
file = "MODEL_NAME.pth"
torch.save(model.state_dict(), file)