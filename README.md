# Convolutional Neural Networks Image Classification (PyTorch)
This project explores the development, training, and evaluation of convolutional neural networks (CNNs) for image classification on the **CIFAR-10 dataset**.

It includes baseline performance with a simple CNN model and two custom-designed CNN architectures that incorporate increasingly advanced features such as deeper layers, dropout, batch normalization, weight decay, and data augmentation.

All the code can be found in `main.ipynb`, which makes use of `pytorch` to implement the model and `matplotlib` to plot the resulting data.


## Dependencies
This project was run in **Google Colab** with GPU acceleration, though it should be possible to set up on local environments as long as all the dependencies are installed.

Make sure to install:
- Python 3.8+
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

```bash
pip install torch torchvision matplotlib numpy
```


## Models Implemented
**SimpleCNN**    
- A two-layer CNN
- Serves as a baseline for comparison.
- Uses ReLU activations, and pooling.


**CustomCNN1**
- A deeper three-layer CNN using:
- Dropout (p = 0.5)
- SGD optimizer
- No data augmentation


**CustomCNN2**
- An advanced six-layer CNN with:
- Batch normalization (optional)
- Dropout (p = 0.6)
- Weight decay (1e-4)
- Adam optimizer
- Extensive data augmentation
  - RandomHorizontalFlip
  - RandomCrop
  - ColorJitter
  - RandomErasing


## Training and Evaluation
Dataset: CIFAR-10

Train/Validation/Test split: 45k / 5k / 10k

Epochs:
- 10 for SimpleCNN and CustomCNN1
- 30 for CustomCNN2

Evaluation Metrics:
- Accuracy (%)
- Validation loss


## Results Summary
**Batch Normalization Disabled**
| Model       | Final Validation Accuracy  | Final Validation Loss  |
|-------------|----------------------------|------------------------|
| SimpleCNN   | 67.29%                     | 0.9295                 |
| CustomCNN1  | 68.19%                     | 0.9106                 |
| CustomCNN2  | 87.07%                     | 0.4215                 |

**Batch Normalization Enabled**
| Model       | Final Validation Accuracy  | Final Validation Loss  |
|-------------|----------------------------|------------------------|
| SimpleCNN   | 69.66%                     | 1.1982                 |
| CustomCNN1  | 75.11%                     | 0.8998                 |
| CustomCNN2  | 88.14%                     | 0.3726                 |


## Notes

All experiments were reproducible using fixed seeds and deterministic training where applicable.

Training and results were carefully logged and compared using both quantitative metrics and visualizations.

More layers, Batch normalization, Adam optimizer, and strong data augmentation had the greatest positive impact on model performance.
