# CNN-LSTM-Violence-detection
This project is a re-implementation of the model described in the paper : https://ieeexplore.ieee.org/document/8852616
# Dependencies
- Python ( I used 3.8.3)
- opencv
- tensorflow 2.x
- keras
# Use 
- make sure you have all the necessary dependencies like Tensorflow 2, Keras, numpy, opencv, especially cuda tools for gpu support as the process is computationally heavy. 
- Clone the project and download the trained weights and put them in the same directory (you can put them wherever you want but then you must add the path for the weights in the "violencemodel.py" file).
- Open a terminal or command prompt if you are on windows and deploy using a command like : 
 ``` 
 python predict_video2.py --input input_path/violence_video.mp4 --output output_path/results.avi --size 128
 ```
- keep in mind that the input and output arguments are required. 
# script at work : 
the model takes 30 frames as an input :
the first 2 videos are obtained by using a rolling average and in the other 2 we only showcase the first frame the model takes every loop run.
<p align="center">
  <img src="https://github.com/souhaiel1/CNN-LSTM-Violence-detection/blob/main/violence-detction%20(1).gif" width="650" height="400" />

## Trained weights : https://drive.google.com/file/d/1phOcWnglLZxsly7gKBab8XJuKzFf5vdp/view?usp=sharing

## Datasets : 
- Movies Fight Detection Dataset :  https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635
- Hockey Fight Detection Dataset : https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89
- VIOLENT-FLOWS DATABASE  : 
https://www.openu.ac.il/home/hassner/data/violentflows/
