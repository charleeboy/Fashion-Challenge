import  numpy as np
import matplotlib.pyplot as plt
import hiddenlayer as h1
#import plotly.graph_objs as go
class Visualizer: 
    def __init__(self, epochs):
        self.epochs = epochs
        self.loss_values = []
        self.epoch_values = {}
    
    #FUNCTION TO APPEND TO A DICTIONARY
    def append_value(self, dict_obj, key, value):
        if key in dict_obj:
            if not isinstance(dict_obj[key], list):
                dict_obj[key] = [dict_obj[key]]
            dict_obj[key].append(value)
            return dict_obj
        else:
            dict_obj[key] = {}
            dict_obj[key] = value
            return dict_obj
    
    #GENERIC FUNCTION TO ADD OTHER DATA TO A SPECIFIC EPOCH
    def add_info(self, epoch, value, string):
        temp_dicht = {}
        if epoch in self.epoch_values:
            self.epoch_values[epoch] = self.append_value(
                self.epoch_values[epoch], string, value)
        else:
            self.epoch_values = self.append_value(self.epoch_values,
                                                  epoch, self.append_value(temp_dicht, string, value))
        return
    
    #ADD LOSS TO A SPECIFIC EPOCH
    def add_loss(self  , epoch, value):
        temp_dicht = {}
        if epoch in self.epoch_values:
            self.epoch_values[epoch] = self.append_value(self.epoch_values[epoch], 'Loss_values', value)
        else:
            self.epoch_values = self.append_value(self.epoch_values,
            epoch, self.append_value(temp_dicht, 'Loss_values', value))
        return
    
    #FUNCTION TO CALL AT THE END OF EACH EPOCH
    def add_trainAccurancy(self, epoch, value):
        self.add_info(epoch, len(
            self.epoch_values[epoch]['Loss_values']), 'Loss_lenght')
        self.add_info(epoch, value, 'Train_Accurancy')
        self.add_info(epoch, np.mean(
            np.array(self.epoch_values[epoch]['Loss_values'])), 'Loss_mean')
        return     
    
    def add_validationAccurancyLoss(self, epoch, value_accurancy, value_loss):
        self.add_info(epoch, value_accurancy, 'Validation_Accurancy')
        self.add_info(epoch, value_loss, 'Validation_Loss')
    #RETURN THE WHOLE DICTIONARY WITH THE DATA
    
    def get_epochs(self):
        return self.epoch_values
    
    #RETURN A SPECIFIC EPOCH
    def get_epoch(self, epoch):
        return self.epoch_values[epoch]
    
    #VIEW CLASSIFY
    def view_classify(img, ps, version="MNIST", lenght = 784 ):
        
        ps = ps.data.numpy().squeeze()

        fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
        ax1.imshow(img.resize_(1, np.sqrt(lenght),
                               np.sqrt(lenght)).numpy().squeeze())
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        if version == "MNIST":
            ax2.set_yticklabels(np.arange(10))
        elif version == "Fashion":
            ax2.set_yticklabels(['T-shirt/top',
                                'Trouser',
                                'Pullover',
                                'Dress',
                                'Coat',
                                'Sandal',
                                'Shirt',
                                'Sneaker',
                                'Bag',
                                'Ankle Boot'], size='small')
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)

        plt.tight_layout()
    
    #DISPLAY THE IMAGE OF THE IMAGE PASSED 
    def imshow(image, ax=None, title=None, normalize=True):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))

        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticklabels('')
        ax.set_yticklabels('')

        return ax

    #GET BASIC PLOT VARIATION LOSS AND ACCURANCY TRAIN AND VALIDATION
    def get_basicLossAccurancyPlot(self):
        accurancy = []
        val_accurancy = []
        loss = []
        val_loss = []
        for i in range(1, self.epochs):
            accurancy.append(self.epoch_values[i]['Train_Accurancy'])
            val_accurancy.append(self.epoch_values[i]['Validation_Accurancy'])
            loss.append(self.epoch_values[i]['Loss_mean'])
            val_loss.append(self.epoch_values[i]['Validation_Loss'])
        print(val_loss, loss)
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.set_title('Model Accurancy')
        ax1.plot(range(1, self.epochs), accurancy, label = 'train')
        ax1.plot(range(1, self.epochs), val_accurancy, label = 'validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('accuracy')
        ax1.legend(loc='upper right')
        ax2.set_title('Model Loss')
        ax2.plot(range(1, self.epochs), loss, label = 'train')
        ax2.plot(range(1, self.epochs), val_loss, label = 'validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('accuracy')
        ax2.legend(loc='upper right')
        
        
        
        plt.show()

    

    
