import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from Model import UNet
from customDataset import RunDataset
import matplotlib.pyplot as plt

def runModel(sample_dir):

    # These are the transforms applied to the validation data. Because the model is not being trained, we don't need to flip and rotate the images. 
    run_transformations = v2.Compose(
            [v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=[512,512]),
            v2.Normalize(mean=[0.0], std=[1.0])]
    )

    # Initialize the dataset with the custom dataset
    dataset = RunDataset(sample_dir=sample_dir, transform=run_transformations)

    # Define test_loader
    batch_size = 1
    validation_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enter the name of your model to run
    file = 'YOUR_MODEL_NAME.pth'
    loaded_model = UNet().to(device)
    loaded_model.load_state_dict(torch.load(file))
    loaded_model.eval()

    predicted_masks = []

    with torch.no_grad():
        for i, (sample, label) in enumerate(validation_loader):
            sample = sample.to(device)
            label = label.to(device)

            output = torch.sigmoid(loaded_model(sample))
            preds = (output > 0.5).float()

            predicted_masks.append(output)
    
    return predicted_masks

if __name__ == '__main__':
    sample_dir = 'PATH TO SAMPLES'
    predicted_masks = runModel(sample_dir)

    batch1 = predicted_masks[0]
    batch1 = torch.Tensor.cpu(batch1)

    plt.figure(figsize=(7, 7))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(batch1[i][0], cmap='gray')
        plt.title(f'Mask {i+1}')

