# ASL_to_Audi

 THis is an flask application to predicit sign language and convert it into an audio.
 

requirement 
  flask 
  opencv-python
  numpy 
  mediapipe 
  keras
  deep_translator
  gtts
  
  We propose a computer-vision system for the task of sign language recognition. Our proposed method doesn’t depend on using glove-based sensors because hand gestures are only a part of sign language. Instead, it captures the hand, 
face, and body motion.

  In this project, for pre-processing the dataset, Mediapipe holistic approach is used to extract key points from a hand gesture, face and pose landmarks. The processed data is then used to train the classification model. Neural network of the pre-trained model 
is later adjusted to new weights. 

  Resultant model is used in later steps for any improvements if necessary. One of the major challenges we faced was the lack of real data, and trusted datasets.
  
  Using OpenCV, the image frames are captured. These are first used for pre-processing, and later for predicting the class labels. Neural network of LSTM model analyses the key points in 
the image frames captured and identifies the possible class labels. 

  Class label with higher probability is the expected result. Identified class label is primarily displayed in English. 
  
  Then the above mentioned class label is converted to the language initially selected using “Deep translator” and "gTTS" Python Libraries and its corresponding 
audio is played.

  INSTRUCTIONS:
    1. Run App.py fie.
    2. Check whether required packages are imported.
    3. Once server is deployed successfully open browser and search for localhost:5000 which will load a web application.
    4. Predicitions can be made once the web application is loaded.
    
