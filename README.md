# secucam
Camera Surveillance System

1) It is recommended to set up the application in a virtual environment.
2) It is also recommended to use VSCode as the development environment to run the application.
3) Download the secucam directory, as well as the requirements.txt file, from this GitHub repository.
4) Inside your virtual environment, run the command: pip install requirements.txt
5) If your PC runs on an Nvidia GPU, then your GPU may be cuda-enabled. This means that YOLOv8 can make use of your GPU to run faster.
   To check your GPU's cuda version, run this command in your PC's CMD prompt:  nvidia-smi
   To check if your GPU is using cuda, run the following python script:
   import torch
   torch.cuda.is_available() # This should return True
   If it returns False then:
   - either your PC does not have a cuda-enabled GPU (run the nvidia-smi command line above to find out)
   - or the PyTorch version installed from the requirements.txt may not be compatible with your PC's cuda version.
     In this case, visit https://pytorch.org/get-started/locally/ to select the version that is compatible with your system.
   If the returned value is still False, that is okay. You can still run the application on your CPU. Just make sure that the CPU version of PyTorch is installed.
6) 
