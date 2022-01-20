# Variational Cycle-GAN model for Emotion Conversion

Tensorflow implementation of [VCGAN](https://www.isca-speech.org/archive/interspeech_2020/shankar20c_interspeech.html) using momenta parameterization of diffeomorphic registration as an intermediary for prosody (F0 and Energy) manipulation. 

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

The above code will create a new file momenta <fraction.mat> at the save_dir location. The feature extraction currently uses DTW-alignment between source and target emotional audio files but it is not a strict requirement. The other parameters that can be played with are the n_mfc,  n_frames, window_len (hard-coded) and window_stride (hard-coded) in the generate_features.py file. 

We now have the complete data pair (source and target emotion) required for training the model.

## Neural Network architecture for Conversion
![Alt text](images/architecture.png?raw=true "Title")
We use fully convolutional neural network for the sampling block of the generator and the discriminator. The diffeomorphic registration is carried out by an RNN-block having fixed/non-learnable parameters. 

## Training the Encoder-Decoder-Predictor model
```
python3 train.py --emo_pair <neu-ang/neu-hap/neu-sad> --train_dir <directory containing training momenta_.mat data> --model_dir <directory to save trained model> 
```
Hyperparameters like learning rate, minibatch-size, #epochs, etc can be modified in the train.py file. To modify the architecture of neural networks, check out the nn_models.py file. It contains the description of neural nets for encoder, decoder and predictor (formerly generator). 

model.py defines a class that creates all the necessary placeholders, variables and functions to use for training and testing. It also generates the summaries which can be visualized using tensorboard module. 

Note: There will be a separate model for every pair of emotion that the corpus contains.  

![Alt text](images/training_validation.png?raw=true "Title")
![Alt text](images/example_pitch.png?raw=true "Title")
## Testing the model

To convert a set of audio files (.wav) from one emotion to another, you need to load the appropriate emotion-pair model and provide path to the data directory. 
```
python3 convert.py --emo_pair <neu-ang/neu-hap/neu-sad> --model_path <complete path to .ckpt file> --data_dir <directory containing .wav files for conversion> --output_dir <directory for saving the converted files> 
```

## Demo of momenta based diffeomorphic registration
![Alt text](images/warping.gif?raw=true "Title")

## Further links
[Download the pre-trained model here](https://drive.google.com/file/d/17EEFnz6-RzmIZn9xqCkCn0yh0Ny5wc6R/view?usp=sharing)

[Demo Audio samples are available here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rshanka3_jh_edu/EUsHjcLhFPpKhX7hkyh2CnIBsZ7Sf14BeTniMA2cGqt_Gw?e=uxaD3b)

[Link to the VESUS dataset](https://engineering.jhu.edu/nsa/vesus/)
