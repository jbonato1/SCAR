# SCAR: Selective-distillation for Class and Architecture-agnostic unleaRning
[Jacopo Bonato](https://scholar.google.com/citations?user=tC1GFkUAAAAJ&hl=it&authuser=1),[Marco Cotogni](https://scholar.google.com/citations?user=8PUz5lAAAAAJ&hl=it), [Luigi Sabetta](https://scholar.google.com/citations?view_op=list_works&hl=en&user=rQBQQjMAAAAJ)


<!--The paper is available on  [![arxiv](https://img.shields.io/badge/arXiv-red)]()-->

## Overview

 SCAR is a novel model-agnostic unlearning algorithm named Selective-distillation for Class and Architecture-agnostic unleaRning. SCAR utilizes metric learning and knowledge distillation techniques to efficiently remove targeted information from models without relying on a retain set. By leveraging the Mahalanobis distance, SCAR shifts feature vectors of instances to forget towards distributions of samples from other classes, facilitating effective metric learning-based unlearning. Additionally, SCAR maintains model accuracy by distilling knowledge from the original model using out-of-distribution images.
![Time](imgs/fig1.png)
 Key contributions of this work include the development of SCAR, which achieves competitive unlearning performance without retain data, a unique self-forget mechanism in class removal scenarios, comprehensive analyses demonstrating efficacy across different datasets and architectures, and experimental evidence showcasing SCAR's comparable or superior performance to traditional unlearning methods and state-of-the-art techniques that do not use a retain set.

## Getting Started


### Installation

```bash
# Clone the repository
git https://github.com/jbonato1/SCAR

# Navigate to the project directory
cd your-repo

# Installation WITH DOCKER

#Step 1:

#Build the docker image from the Dockerfile : 
docker build -f Dockerfile -t scar:1.0 . 

#Step 2:

#Run your image : 
docker run -it --gpus all -v "/path_to_dataset_folder":/root/data -v "/path_to_scar_folder":/scar scar:1.0 /bin/bash

# Install LOCALLY 
pip install -r requirements.txt
```

## Code Execution
TO DO