import torch
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from .Model import UNet
from .customDataset import RunDataset
import matplotlib.pyplot as plt

def runModel(sample_dir, model_name):

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
    run_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    file = model_name
    loaded_model = UNet().to(device)
    loaded_model.load_state_dict(torch.load(file, map_location=device))
    loaded_model.eval()

    predicted_masks = np.empty((len(run_loader), 512, 512), dtype=np.uint8)

    with torch.no_grad():
        for time_point, sample in enumerate(run_loader):
            sample = sample.to(device)

            output = torch.sigmoid(loaded_model(sample))
            prediction = (output > 0.5).float()
            prediction = prediction[0][0].numpy().astype(np.uint8)

            predicted_masks[time_point] = prediction

            print(f'{time_point+1} of {len(run_loader)}')
    
    return predicted_masks


# # Test to make sure the function and the model as a whole are working properly
# if __name__ == '__main__':
#     sample_dir = 'ML_Model/test'
#     model_name = 'ML_Model/lesion_finder_v1.pth'
#     predicted_masks = runModel(sample_dir, model_name)


#     # This little block of code is to get the first sample image. It's inefficient but I'm lazy
#     run_transformations = v2.Compose(
#             [v2.ToImage(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Resize(size=[512,512]),
#             v2.Normalize(mean=[0.0], std=[1.0])]
#     )
#     test_dataset = RunDataset(sample_dir, run_transformations)
#     test_loader = DataLoader(test_dataset, 1, False)
#     test_sample = iter(test_loader)
#     sample = next(test_sample)

#     batch1 = predicted_masks[0]
#     batch1 = torch.Tensor.cpu(batch1)

#     fig, ax = plt.subplots()
#     ax.imshow(sample[0][0], alpha=1.0)
#     ax.imshow(batch1[0][0], alpha=0.1, cmap='gray')
#     ax.axis('off')
#     plt.title('First Sample + First Prediction')
#     plt.show()


