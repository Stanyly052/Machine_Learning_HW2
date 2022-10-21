Steps to run the code:
    Step 1: Creat the env by conda, the env file "Yuzhi_Wang.yml"
    Step 2:  Run the code below to train the models:
        python HW1_Pytorch.py --lr 0.001 --optimizer SGD --SGD_momentum 0.9 --model L_SVM
        All the parameters above can be changed, e.g.: different LRs, optimizer with momentum (SGD-M), different model (Logistic)
    Step 3: Change the file path in "Plot-Function_Smart.py", then you can draw all plots with the code below:
        python Plot-Function_Smart.py