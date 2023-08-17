This repository contains code and dataset used in the paper, [_Auto-Gait: Automatic Ataxia Risk Assessment with Computer Vision from Gait Task Videos_](https://dl.acm.org/doi/10.1145/3580845)

Dataset: https://rochester.box.com/s/k9lh33m98mx2hum3kyp2c4j6eoo6bcde 

- Ataxia is a degenerative disease of the nervous system that surfaces as difficulty with motor control, such as walking imbalance. Many patients with Ataxia do not have easy access to neurologists – especially those living in remote localities and developing/ underdeveloped countries. 
- In a multi-year collaboration effort with READISCA, we collected a video dataset of ataxia diagnosed and control subjects from 11 clinics located in 8 different states across the United States.
- The dataset contains 150 fully anonymized 6 seconds long videos from 89 unique study participants (24 control, 65 diagnosed) performing gait task.
- We developed a computer vision and machine learning pipeline to identify, track, and separate participants from complex surroundings and predict their risk and severity of Spinocerebellar Ataxia.
- Ataxia risk-prediction model achieves 83.06% accuracy and an 80.23% F1 score. 
- Ataxia severity-assessment model achieves MAE score of 0.6225 and a Pearson's correlation coefficient score of 0.7268.
- Our model performs competitively while tested on clinics completely unseen during training.
- Our feature importance analysis shows the model automatically picks up traits that are consistent with established clinical knowledge.


Citation:

```
@article{10.1145/3580845,
author = {Rahman, Wasifur and Hasan, Masum and Islam, Md Saiful and Olubajo, Titilayo and Thaker, Jeet and Abdelkader, Abdel-Rahman and Yang, Phillip and Paulson, Henry and Oz, Gulin and Durr, Alexandra and Klockgether, Thomas and Ashizawa, Tetsuo and Investigators, Readisca and Hoque, Ehsan},
title = {Auto-Gait: Automatic Ataxia Risk Assessment with Computer Vision from Gait Task Videos},
year = {2023},
issue_date = {March 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {7},
number = {1},
url = {https://doi.org/10.1145/3580845},
doi = {10.1145/3580845},
abstract = {Many patients with neurological disorders, such as Ataxia, do not have easy access to neurologists, -especially those living in remote localities and developing/underdeveloped countries. Ataxia is a degenerative disease of the nervous system that surfaces as difficulty with motor control, such as walking imbalance. Previous studies have attempted automatic diagnosis of Ataxia with the help of wearable biomarkers, Kinect, and other sensors. These sensors, while accurate, do not scale efficiently well to naturalistic deployment settings. In this study, we propose a method for identifying ataxic symptoms by analyzing videos of participants walking down a hallway, captured with a standard monocular camera. In a collaboration with 11 medical sites located in 8 different states across the United States, we collected a dataset of 155 videos along with their severity rating from 89 participants (24 controls and 65 diagnosed with or are pre-manifest spinocerebellar ataxias). The participants performed the gait task of the Scale for the Assessment and Rating of Ataxia (SARA). We develop a computer vision pipeline to detect, track, and separate the participants from their surroundings and construct several features from their body pose coordinates to capture gait characteristics such as step width, step length, swing, stability, speed, etc. Our system is able to identify and track a patient in complex scenarios. For example, if there are multiple people present in the video or an interruption from a passerby. Our Ataxia risk-prediction model achieves 83.06\% accuracy and an 80.23\% F1 score. Similarly, our Ataxia severity-assessment model achieves a mean absolute error (MAE) score of 0.6225 and a Pearson's correlation coefficient score of 0.7268. Our model competitively performed when evaluated on data from medical sites not used during training. Through feature importance analysis, we found that our models associate wider steps, decreased walking speed, and increased instability with greater Ataxia severity, which is consistent with previously established clinical knowledge. Furthermore, we are releasing the models and the body-pose coordinate dataset to the research community - the largest dataset on ataxic gait (to our knowledge). Our models could contribute to improving health access by enabling remote Ataxia assessment in non-clinical settings without requiring any sensors or special cameras. Our dataset will help the computer science community to analyze different characteristics of Ataxia and to develop better algorithms for diagnosing other movement disorders.},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = {mar},
articleno = {26},
numpages = {19},
keywords = {datasets, computer vision, pose estimation, gait, ataxia}
}
```
