# Anomaly Detection in Smart Buildings
### using Federated Learning
                                     
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Pathi-rao/Strive_Capstone">
    <img src="https://github.com/Pathi-rao/Strive_Capstone/blob/main/Figures/main.PNG" alt="Logo">
  </a>
  <br />



## Table of Contents

* [Introduction](#introduction)
* [Inspiration](#inspiration)
* [Setup](#setup)
* [What it does](#what-it-does)


    
## Introduction
Smart buildings are becoming a reality faster than everyone thought. AI is being integrated into most of them to make them even smarter from "Heating Ventilation AC" (HVAC) management, parking assitance to much more. As these buildings gets more and more sophisticated, the communication between different components increase within the network and they are more susceptible to external threats like cyber threats and data breach. So, how can we protect them from such anamolies in the network?
    
## Inspiration
- My first inspiration comes from lack of data for most of the problems in AI. Federated Learning addresses this issue by training through shared data by which everyone can be benifited.
- Secondly, [openmined](https://www.openmined.org/) and their initiative towards privacy preserving techniques whose goal is where "People and organizations can host private datasets, allowing data scientists to train or query on data they *cannot see*. The data owners retain complete control: data is never copied, moved, or shared."
    
## Setup
1. Clone the repo using the following command:
```bash
git clone https://github.com/Pathi-rao/Strive_Capstone.git 
```
2. Create a virtual environment with conda and install the packages in requirements file
```bash
conda create --name <envname> --file requirements.txt
```	
3. Open 3 different termnials and activate the conda environment using
```bash
conda activate <envname>
```
4. `cd` into the project directory and run the `server` script in one of the terminals and run `client` script in the other *two* terminals
```bash
python server.py
```
```bash
python client.py
```

## What is does
- After successfully executing the scripts, both server and clients will get initialized. A [flower](https://flower.dev/) server will start and trains for specified number of training rounds. For each rounds, the server sends the model to the client where it will be trained with it's local data and saved. The trained model will be sent back to the server, get's aggregated and starts the process once again.
- After each round, both the loss and accuracy will also be displayed to better understand how the model will get improved overtime.
    
    

