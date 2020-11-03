#  road segmentation 
### overall introduction



  - data:read,augment and process the original data 
  - model:six different models:
    segnet,unet,unet_dillated,unet_attention,linknet,linknet_dillated,linknet_attention
  - train_predict:get the h5file of weights of model
  - run:get the final submission
  - metrics: calculate tne F1-socore and accuracy
  - loss: dice-loss is used

### reproduce the results


  - fistly, install the required libiraries in requirements.txt via pip
  - Then,considering the large memory of the file of weights,we only provide the weights file submitted to crowdAI,which belongs to the linknet_dillated
    ~~~shell
    python run_submission.py
  - If you want to check the results of other models:
    ~~~~shell
     python train_predict.py
    ~~~~
    ~~~~shell
     python run.py
    ~~~~
  - the file of train_predict.py will help you get all the models'weightes files and some csvfiles to record the relavent data of every epoch. The run.py can make you choose the modle you want to create submission when you choose the method "1"


You can also:
  - Import and save files from GitHub, Dropbox, Google Drive and One Drive
  - Drag and drop markdown and HTML files into Dillinger
  - Export documents as Markdown, HTML and PDF

et a feel for Markdown's syntax, type some text into the left window and watch the results in the right.





