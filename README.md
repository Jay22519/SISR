# final-year-project 

## This branch contains the backend for SISR developed using FastAPI 

#### Steps to run 
          1) Make sure you have correct models in "Pretrained Model" and "Upscaling methods" directory 
          2) Install requirements.txt
          3) Now run python main.py 
          4) Once the Server starts you can go to localhost/predict in your browser and then can add Images/Videos and relevant parameters required for Upscaling
          5) Click on submit button and wait for some time to get results . Your result (wheter it is Image or Video) will be stored in your main directory by the name of Output.extension ( jpg in case of Image and mp4 in case of Video)
          
          
#### Parameters 
        1) First video is for factor , i.e by how much factor you want to upscale your Image ( should be a factor of 2) 
        2) Second field is for type_input , i.e , 1 for Video input and 0 for Image 
        3) Third is whether audio enhancement is disabled or enabled ( 1 for enabled and 0 for disabled) 
        
 
