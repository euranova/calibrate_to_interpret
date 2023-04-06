
Calibrate to interpret 
======================

Install libraries
------------------ 

```
    pip install -r requirements.txt
```
Install dirichlet_python
------------------ 
```
cd code final
# Clone the repository
git clone git@github.com:dirichletcal/dirichlet_python.git
# Go into the folder
cd dirichlet_python
# Create a new virtual environment with Python3
python3.8 -m venv venv
# Load the generated virtual environment
source venv/bin/activate
# Upgrade pip
pip install --upgrade pip
# Install all the dependencies
pip install -r requirements.txt
pip install --upgrade jaxlib
```

Reproduce the experiments  :
---------------------------


First manually define variable (device, model, interpretation method ...) in the configurations files which are :
- code_final/cifar100/conf.ini for experiments on cifar100 
- code_final/food101/conf.ini for experiments on food101



All the possible choices for each variable are specified in the conf.ini files  
The Experiments jupyter Notebook shows how to use manually each step of the experiment.  
Additionally we provide the code to run the whole experiment, take into account that this code is long to run.  
Run the following command :

Cifar100 :

```
    python -m code_final.cifar100.main_exp
```
The dataset will be loaded automatically in code_final/src/dataset  
The models can be downloaded here  :  
	resnet 32 : https://drive.google.com/file/d/1qnPD1DH6Dy94Z4Pmh1cvc22aPwFArx8G/view?usp=sharing   
	vgg16 :  https://drive.google.com/file/d/1j0Y0BpiImKC8ZT9JX6xB6b5-8gQlWf9Y/view?usp=sharing   

Food101 :  

```
    python -m code_final.food101.main_exp
```
Dataset archive need to be downloaded at :  https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/  
The resnet50 model need to be downloaded at : https://drive.google.com/file/d/17miH6qv6kQwCqJhiIZVWOh-quDtQQSIh/view?usp=sharing  
then extract the images folder into code_final/food101/  



For both dataset, the directory's architecture is as follow :   
    &nbsp;&nbsp;--dataset  
            &nbsp;&nbsp;&nbsp;-results_folder  
            &nbsp;&nbsp;&nbsp;&nbsp;    - img_non_calib  , save interpretation of non calibrated model as npy  
            &nbsp;&nbsp;&nbsp;&nbsp;    - img_calib  , save interpretation of calibrated model as npy   
            &nbsp;&nbsp;&nbsp;&nbsp;    - auc_non_calibrated ,  save deletion area of non calibrated model as npy   
            &nbsp;&nbsp;&nbsp;&nbsp;   - auc_calibrated ,  save deletion area of calibrated model as npy   
            &nbsp;&nbsp;&nbsp;&nbsp;    - auc_random ,  save deletion area of random saliency for both calibrated and uncalibrated model as npy   



Analyse the results  :  
----------------------


In order to obtain  the results and plots that are prensented in the paper, the experiments need to be completly executed.   
Yet the experiments can be stopped at any moment to analyse intermediate results and produce the graphics.   
The full raw results are also saved  as csv. They can be donwloaded at : https://drive.google.com/drive/folders/1RyB0JVbwHXTuxzHzD3mmgpYDroorvA6v?usp=sharing and need to be put at the root of the results folder .   
The Results_analysis jupyter Notebook shows how to reproduce the visualisation from the article.   


First manually define variable (device , model, results folder  ...) in the configurations files which are :   
- code_final/results/conf.ini for plotting the results   

To convert the results fom cifar100/food101 folder to csv  :   

```
    cd code_final/results/
    python results_to_csv.py
```

ATTENTION !! the corresponding raw results will be replaced by the new results.  



To compute total variation  :

```
    cd code_final/results/
    python total_variation.py

```

 

To plot ssim :

```
    cd code_final/results/
    python analyse_ssim.py <model_name> <dataset_name> <calibration_method_name>
```
 


Note : this folder contains the dirichlet_python library  ! 

To cite our work 
ref :  
```
@InProceedings{10.1007/978-3-031-26387-3_21,
author="Scafarto, Gregory
and Posocco, Nicolas
and Bonnefoy, Antoine",
editor="Amini, Massih-Reza
and Canu, St{\'e}phane
and Fischer, Asja
and Guns, Tias
and Kralj Novak, Petra
and Tsoumakas, Grigorios",
title="Calibrate to Interpret",
booktitle="Machine Learning and Knowledge Discovery in Databases",
year="2023",
publisher="Springer International Publishing",
address="Cham",
pages="340--355"
}
```





