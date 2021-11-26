import matplotlib.pyplot as plt

def plot_loss(history):
    history['agent_avg_loss'].plot(label='avg_loss');
    history['agent_avg_loss'].rolling(10).mean().plot(label='rolling(10) mean of avg_loss');
    plt.title('Agent average loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend();

def plot_score(history):
    history['score'].plot(label='score');
    history['score'].rolling(100).mean().plot(label='rolling(100) mean of score')
    plt.title('Agent score')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend();