import numpy as np
import matplotlib.pyplot as plt


def one_hot(num, length):
    ans = np.zeros(length)
    ans[num] = 1
    return ans.reshape(-1,1)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
        
    return output

def get_accuracy(answers, predictions):
    
    max_in_columns = np.max(predictions, axis = 0)
    one_hot_predictions = (predictions == max_in_columns).astype(int)
    
    mistake_count = np.any(one_hot_predictions != answers, axis=0)
    accuracy = 1 - np.sum(mistake_count)/answers.shape[1]
    
    print(f"Accuracy on {answers.shape[1]} samples is {accuracy}")
    return accuracy


def show_mistakes(answers, predictions):
    pass   

def show_and_predict(network, X, answers, index):
    
    """
    Just for one sample
    input.shape must be (., 1) for correct showing
    
    """
    output = X[:, index].reshape(-1,1)
    
    image_28x28 = output.reshape(28, 28)
    plt.imshow(image_28x28, cmap='gray')
    plt.axis('off') 
    plt.show()
    
    for layer in network:
        output = layer.forward(output)
    
    prediction = np.argmax(output)
    true_label = np.argmax(answers[:, index].reshape(-1,1))
    
    print(f"Label = {prediction}, probability = {output[prediction]}, true_label = {true_label} ")
    
#     return output