

![Alt text](images/training_validation.png?raw=true "Title")
![Alt text](images/example_pitch.png?raw=true "Title")


# Variational Cycle-GAN model for Emotion Conversion

Tensorflow implementation of our [VCGAN](https://www.isca-speech.org/archive/interspeech_2020/shankar20c_interspeech.html) using momenta parameterization of diffeomorphic registration as an intermediary for prosody (F0 and Energy) manipulation. 


## Neural Network architecture for Conversion
![Alt text](images/architecture.png?raw=true "Title")
We use fully convolutional neural network for the sampling block of the generator and the discriminator. The diffeomorphic registration is carried out by an RNN-block having fixed/non-learnable parameters. 

## Demo of momenta based diffeomorphic registration
![Alt text](images/warping.gif?raw=true "Title")


## Main Dependencies

- Python 3.7 (or higher)
- tensorflow 1.14
- librosa 
- pyworld 

## Feature Extraction

The data directory is organised as:
```
data
├── neutral-angry
│   ├── train
|       ├── neutral (wav files)
|       ├── angry (wav files)
|   ├── valid
|       ├── neutral (wav files)
|       ├── angry (wav files)
|   ├── test
|       ├── neutral (wav files)
|       ├── angry (wav files)
|
├── neutral-happy
│   ├── ...
|
├── neutral-sad
│   ├── ...
```

Extract features (mcep, f0) from each speech file.  The features are stored in pair-wise manner in .mat format (for matlab compatibility). 
```
python3 generate_features.py --source_dir <source (neutral) emotion wav files> --target_dir <target emotion wav files> --save_dir <directory to save extracted features> --fraction <train/valid/test>
```

Unfortunately, the above code only extracts the mel-cepstral and F0 features. We need the intermediate representation (momenta variable). We extract it using the matlab implementation of VLDDMM available in the current folder. The implementation is only available in Matlab (2018 or higher).

Extract the momenta variables required for encoder training.
```
generate_momenta.m <path to data (.mat) files> <fraction (train/valid/test)>
```
The above code will create a new file momenta_<input_file.mat> in the same folder as data path.

We now have the complete data required for training the model :)

## Training the Encoder-Decoder-Predictor model
```
python3 train.py --emo_pair <neu-ang/neu-hap/neu-sad> --train_dir <directory containing training momenta_.mat data> --model_dir <directory to save trained model> 
```
Hyperparameters like learning rate, minibatch-size, #epochs, etc can be modified in the train.py file. To modify the architecture of neural networks, check out the nn_models.py file. It contains the description of neural nets for encoder, decoder and predictor (formerly generator). 

model.py defines a class that creates all the necessary placeholders, variables and functions to use for training and testing. It also generates the summaries which can be visualized using tensorboard module. 

Note: There will be a separate model for every pair of emotion that the corpus contains.  

## Testing the model
To convert a set of audio files (.wav) from one emotion to another, you need to load the appropriate emotion-pair model and provide path to the data directory. 
```
python3 convert.py --emo_pair <neu-ang/neu-hap/neu-sad> --model_path <complete path to .ckpt file> --data_dir <directory containing .wav files for conversion> --output_dir <directory for saving the converted files> 
```

## Further links
[Download the pre-trained model here](https://drive.google.com/file/d/17EEFnz6-RzmIZn9xqCkCn0yh0Ny5wc6R/view?usp=sharing)

[Demo Audio samples are available here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rshanka3_jh_edu/EUsHjcLhFPpKhX7hkyh2CnIBsZ7Sf14BeTniMA2cGqt_Gw?e=uxaD3b)

[Link to the VESUS dataset](https://engineering.jhu.edu/nsa/vesus/)
