Code for paper "MFC-CNN：An Automatic Grading Scheme for Light Stress Levels of Lettuce (Lactuca sativa L.) Leaves"
# architecture
```
|--multi-input
    |--logs            # saving running logs here
    |--model           # saving the trained model parameters here
    |-- results        # 
    |--load_data.py    # generating multi-scale images
    |--main.py         # main file here
    |--pre_single.py   # the code of preparing datas and labels
```
# run
```
python main.py
```
# data
All the image set is provided in  Googledrive [https://drive.google.com/drive/folders/1Mz2YQ-Sm7RL-qa4tHvJ89_4rnoHq-gkw?usp=sharing] and Figshare. In figshare, due to the capacity limitation of each project, we upload the data in two projects. (1) https://figshare.com/articles/figure/Untitled_Item/13106867 (2) https://figshare.com/articles/figure/Part2--Image_set_for_the_paper_MFC-CNN_An_Automatic_Grading_Scheme_for_Light_Stress_Levels_of_Lettuce_Lactuca_sativa_L_Leaves_/13107197
# requirments:
 ```
 python=3.7
 tensorflow-gpu=1.13.1
 keras=2.3.1
 numpy=1.18.0
 scikit-learn=0.22.2
 scipy=1.2.1
```
# cite
updated later