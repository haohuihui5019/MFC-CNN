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
The relative paper should be cited when you use our code or dataset.
```
@article{HAO2020105847,
title = "MFC-CNN: An automatic grading scheme for light stress levels of lettuce (Lactuca sativa L.) leaves",
journal = "Computers and Electronics in Agriculture",
volume = "179",
pages = "105847",
year = "2020",
issn = "0168-1699",
doi = "https://doi.org/10.1016/j.compag.2020.105847",
url = "http://www.sciencedirect.com/science/article/pii/S0168169920313934",
author = "Xia Hao and Jingdun Jia and Wanlin Gao and Xuchao Guo and Wenxin Zhang and Lihua Zheng and Minjuan Wang",
keywords = "Classification, Stress grading, Deep learning, Multiscale input, Cascade operation",
}
```
