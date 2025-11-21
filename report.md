Team 06 Report: Leaf Disease Classification

Repositories: https://github.com/kushagra25aiz0001-design/collaborative_cnn_team06 | https://github.com/gaurav25aiz0005/collaborative_cnn_team06 

Overview:
We collaborated to classify 15 plant diseases using two different datasets. 
User 1 trained a ResNet50 on the smaller PlantVillage dataset.
User 2 trained a MobileNetV2 on the larger Kaggle dataset. We used a custom script to map the specific disease folders to a global index so both models produced compatible outputs.

Performance & Observations:

Model V1 (ResNet50): Performed perfectly on its own small dataset but failed significantly when tested on the Kaggle dataset. It struggled with the different lighting and backgrounds (overfitting).

Model V2 (MobileNetV2): Performed excellent on its own large dataset and surprisingly well on the small dataset too.

Conclusion:
The MobileNetV2 model trained on the larger, more diverse dataset.
It proved that having a diverse dataset is more important for generalization than having a deep, complex model like ResNet. 