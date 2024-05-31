# Generating-Cyber-Grooming-Chats-with-LLMs
Repository for the Individual Research Module at the University of Potsdam on cyber grooming detection in online Chats.
The goal of the project is to classify online chats into predatory and non-predatory using the multilingual model XLMRoBERTa. The training and evaluation of the model were conducted in 3 different settings to asses its ability to distinguish between cyber grooming and non-cyber grooming conversations in English, German, French and Dutch. The model was evaluated additionally using the CheckList framework to test some of its specific capabilities. The dataset used in the project is the publicly available PAN12 dataset, provided by the PAN Lab at the 2012 CLEF conference for a shared task on sexual predator identification.

## Setup 
Install all necessary packages with requirements.txt
```
$ pip install requirements.txt
```
To Run the training
