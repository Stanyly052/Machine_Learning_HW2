import matplotlib.pyplot as plt
import os
import numpy as np
import torch

class plot_model():
    """This code aims to plot all Pytorch-tensors within the targeted directory
        
        Args:
            path_directory: Path of directory
            suffixes_targted: type of file in the output list

        Output:
            Plotted figures
    """

    def __init__(self, 
                 path_directory: str, 
                 suffixes_targted:str = ".pt",
                 Curve_label:str = "LR",
                 Model_Type:str = "SGD & Logistic",
                 ) -> None:
        """Initialize the plot model
        """
        super(plot_model, self).__init__()

        self.path_directory = path_directory
        self.suffixes_targeted = suffixes_targted
        self.Curve_label = Curve_label
        self.Model_type = Model_Type
        
        if type(self.path_directory) == list:
            self.compare_different_models = True
        else:
            self.compare_different_models = False
        
        if self.compare_different_models:
            folder_1_name = self.path_directory[0].split("/")[-1]
            self.optimizer_1_name = folder_1_name.split("_")[0]
            self.model_1_name = folder_1_name.split("_")[1]

            folder_2_name = self.path_directory[1].split("/")[-1]
            self.optimizer_2_name = folder_2_name.split("_")[0]
            self.model_2_name = folder_2_name.split("_")[1]


    def search_All_PT_With_Suffixxes(self):
        """Return a list that includes all pt paths 
        
        Args:
            None

        Returns:
            path_list
        """
        path_list = []
        if type(self.path_directory) == str:
            for file in os.listdir(self.path_directory):
                file_path = os.path.join(self.path_directory, file)
                if os.path.isfile(file_path):
                    if file_path.endswith(self.suffixes_targeted):
                        path_list.append(file_path)
                else:
                    raise Exception("Target file is not recognized")
        elif type(self.path_directory) == list:
            for directory in self.path_directory:
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        if file_path.endswith(self.suffixes_targeted):
                            path_list.append(file_path)
                    else:
                        raise Exception("Target file is not recognized")
        
        return path_list

    def __xAxisRange(self,
                    path_list: list,
                    ):
        """Return the range of x-axis to plot figures

        Args:
            path_list: list

        Returns:
            x_axis_range: int      
        """
        x_range = list(torch.load(path_list[0]).size())[0]
        assert type(x_range) == int

        return list(torch.load(path_list[0]).size())[0]

    def __curve_label(self,
                      file_path:str,
                      ):
        """Return the label of the curve, get this information from the file name

        Args:
            file_path: str : the path of individual file

        Returns:
            label: str      
        """
        parameter_space = file_path.split("/")[-1].split("_")

        if not self.compare_different_models:
            if self.Curve_label == "LR":
                for parameter in parameter_space:
                    if "LR" in parameter:
                        curve_label = parameter
                        assert type(curve_label) == str
        else:
            if self.Curve_label == "LR":
                for parameter in parameter_space:
                    if "LR" in parameter:
                        curve_label = parameter
                        assert type(curve_label) == str

        return curve_label

    def __model_name(self,
                    file_path:str,
                    ):
        """Extract the type of model from file name

        Args:
            file_path : str : the path of individual file

        Returns:
            model_name : str        
        """
        parameter_space = file_path.split("_")
        for parameter in parameter_space:
            if "SVM" in parameter:
                model_name = "L_" + parameter
                assert type(model_name) == str
            elif "Logistic" in parameter:
                model_name = parameter
                assert type(model_name) == str

        return model_name.split(".")[0]

    def __optimizer_name(self,
                    file_path:str,
                    ):
        """Extract the type of optimizer from file name

        Args:
            file_path : str : the path of individual file

        Returns:
            optimizer_name : str        
        """
        parameter_space = file_path.split("_")
        for parameter in parameter_space:
            if parameter == "SGD" or parameter == "SGD-M":
                return parameter

    def Plot_AllFiles_InPathList(self,
                                 ):
        """Plot all tensors and save a .png figure 
        
        Args:
            None
        
        No Return:
            png figure
        """
        path_list = self.search_All_PT_With_Suffixxes()
        x_range = self.__xAxisRange(path_list)
        x_axis = np.arange(0, x_range, 1)
        
        plt.clf()
        plt.figure()
        plt.yscale("linear")
        for file_path in path_list:
            y_data = torch.load(file_path).detach().numpy()
            label = self.__curve_label(file_path)
            plt.plot(x_axis, y_data, 'o-', label=label)
            plt.legend()
        plt.xlabel("Epoch Number")
        plt.ylabel("Train Loss")
        model_name = self.__model_name(path_list[0])
        optimizer_name = self.__optimizer_name(path_list[0])
        plt.title("{} & {}".format(model_name,
                                   optimizer_name,      
                                  )
                 )
        plt.savefig("{}&{}.png".format(model_name,
                                       optimizer_name,  
                                      )
                   )

        


if __name__ == "__main__":

    path_list = "/home/yuzhi/ML_HW_2/exp_result/SGD-M_Logistic"

    suffix = ".pt"
    plot_model(path_list, suffix).Plot_AllFiles_InPathList()




