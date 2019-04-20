import numpy as np
import matplotlib.pyplot as plt

# =================================================*Functions*==========================================================


def resultplot(results):
    plt.figure()
    plt.plot(results.epoch, np.array(results.results['tst']), label='Test Accuracy')
    plt.plot(results.epoch, np.array(results.results['train']), label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

