# team-linden-p2# Project 2 CSCI-8360 : Cilia Segmentation
## Team-linden

## Member (Ordered by last name alphabetically)
* Abolfazl Farahani (a.farahani@uga.edu)
* Jonathan Myers (submyers@uga.edu)
* Jiahao Xu (jiahaoxu@uga.edu
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

### Convolutional Neural Network (Tiramisu) version2

In many respects, data science serves as successful fields for rapid algorithmic advances due to the intrensic diviersity in approximation problems. To put things another way, the many data analysis problems carry different expectations regarding accuracy, flexibility, and scalability, meaning the number of cases for conditional optimization expands distinct problems exponentially. One, of many, such adaptions from the University of Montreal [Simon Jegou] is the One Hundread Layers Tiramisu model, an extension of the Densely Connected
Convolutional Networks (DenseNets) model [Gao Huang]. 

Simon Jegou supplies a Github copy of the program, but we found a fairly useful Github packages from Brendan Fortuner (https://github.com/bfortuner/pytorch_tiramisu), Zujun Deng (https://github.com/ZijunDeng/pytorch-semantic-segmentation), and Maulik Shah (https://github.com/whusym/Cilia-segmentation-pytorch-tiramisu). Please visit our team's Wiki website to view detailed analysis of the functions supplied and how we applied them to cilia frames.

## References

[Simon Jegou] Simon Jegou, Michal Drozdzal, David Vazquez, Adriana Romero1, and Yoshua Bengio1, "The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation", arXiv:1611.09326v3

[Gao Huang] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger, "Densely Connected Convolutional Networks", arXiv:1608.06993





