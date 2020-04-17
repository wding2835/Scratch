import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def main():
    x = np.linspace(-5,5,100)

    y = sigmoid(x)

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x,y,'r')

    # show the plot
    plt.show()





if __name__ == '__main__':
    main()