import matplotlib.pyplot as plt
import seaborn as sns


def plot_experiment_stat(df, key='score', window=100, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    df[key].plot(ax=ax, label=key)
    df[key].rolling(window).mean().plot(ax=ax, label=f'Rolling {window} mean {key}')
    plt.legend()
    if title is None:
        title = f'Agent\'s {key}'
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(key)


def plot_agent_stats(df, stats=None):
    sample = df['agent'][0]
    if stats:
        keys = [k for k in sample.keys() if k in stats]
    else:
        keys = sample.keys()
    if len(keys) > 6:
        rows, cols = 2, len(keys) // 2
        height = 6
    else:
        rows, cols = 1, len(keys)
        height = 4
    fig, ax = plt.subplots(rows, cols, figsize=(15, height))
    ax = ax.reshape(-1, )
    for k, axis in zip(keys, ax):
        series_k = df['agent'].transform(lambda d: d[k])
        series_k.plot(ax=axis)
        axis.set_title(k)
        axis.set_xlabel('Episode')
    plt.tight_layout()


def plot_evaluation_scores(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    sns.stripplot(x='scores', data=df, ax=axes[0, 0]);
    sns.boxplot(x='scores', data=df, ax=axes[0, 1])
    sns.stripplot(x='moves_mse', data=df, ax=axes[1, 0]);
    sns.boxplot(x='moves_mse', data=df, ax=axes[1, 1])
    plt.suptitle('Evaluation charts')
    plt.tight_layout()
