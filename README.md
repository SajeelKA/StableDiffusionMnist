# StableDiffusionMnist
Stable Diffusion Model training from scratch using MNIST dataset on Google Colaboratory

![3](https://github.com/SajeelKA/StableDiffusionMnist/blob/main/images/result3.png)
![5](https://github.com/SajeelKA/StableDiffusionMnist/blob/main/images/result5.png)

# Setup Instructions
This project is designed to work on Google Colaboratory, so there are references in the code to "mount" to the Google Drive folders to access the required files and folders. 

Therefore, if you want to run the project as-is, you will need to make the root dir in Google Drive as '/content/drive/MyDrive/allRepos/', then upload this project within this root dir in Google Drive.

# Instructions for running training
The training can be run by running the cells in the "train.ipynb" notebook. The customizable hyperparameters available are:
  
  "--batchSize", required = False, type=int, default = 32 (batch size for training)
    
  "--learningRate", required = False, type=float, default = 1e-4 (learning rate for training)
  
  "--epochs", required = False, type=int, default = 10 (number of times to pass over mnist dataset for training)
    
  "latentEmbedChannels", required = False, type=int, default = 4 (output channels for VAE encoder before going to UNET denoiser)
    
  "--textEmbedChannels", required = False, type=int, default = 16 (embedding size for digit that will be input as label)

  "--timeSteps", required = False, type=int, default = 300   (number of alpha and beta coefficient timesteps for noise scheduling)
    
  "--usedProcessor", required = False, default = 'cpu' (whether to use GPU or plain CPU for training)
    
  "--savePath", required = True, default = None (where to save model every 5 epochs)

# Instructions for running inference
The inference can be run by running the cells in the "inference.ipynb" notebook. The customizable hyperparameters available are:
  
  "--fileName", required = False, type=int, default = 'testStableDiffusion.pth' (file name of model)
    
  "--digit", required = False, type=float, default = 3 (which digit to draw using the pretrained stable diffusion model)
    
  "--rootDir", required = False, default = '/content/drive/MyDrive/allRepos/Diffusion/' (directory of the current project)
    
  "--batchSize", required = False, type=int, default = 4 (batch size for output digits)
    
  "--embedChannels", required = False, type=int, default = 16 (embedding channels for digit input (will be dependent on the architecture of pretrained model being used for inference))
  
  "--timeSteps", required = False, type=int, default = 300 (how many timesteps to denoise image)  
  
  "--conditionalScale", required = False, type=float, default = 5.0 (higher scale should mean output samples are closer to input, while lower scale gives more flexibility/originality to the output image)

