# team-linden-p2# Project 2 CSCI-8360 : Cilia Segmentation
## Team-linden

## Member (Ordered by last name alphabetically)
* Abolfazl Farahani (a.farahani@uga.edu)
* Jonathan Myers (submyers@uga.edu)
* Jiahao Xu (jiahaoxu@uga.edu)

## Mehtods
### Pixel Variance

### Optical Flow

### Convolutional Neural Network (Tiramisu)
Our neural network codes are originate from https://github.com/bfortuner/pytorch_tiramisu.
- model/layers.py, model/tiramisu.py are basically all the same as the origianl repo, we only modified the dropout rate from 0.2 to 0.1
- /dataset/joint_transform.py is the same, expect we 
-- changed the rate of randomized crop area from [0.08, 1] to [0.45, 1]
-- changed the rate of the aspect ratio from [3/4, 4/3] to [0.5, 2]
- utils/training.py is the same, expect we added a test image prediction function, get_test_results
- ./dataset/cilia.py makes the cilia data to be appropriate fitted to the tiramisu model

Also, we got some inspiration from https://github.com/whusym/Cilia-segmentation-pytorch-tiramisu
to make the dataset and feed to the model.
