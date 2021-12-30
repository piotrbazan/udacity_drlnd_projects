import matplotlib.pyplot as plt


def plot_scores(df, window=100):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    df['score'].plot(ax=ax, label='score')
    df['score'].rolling(window).mean().plot(ax=ax, label=f'Rolling {window} mean score')
    plt.legend()
    plt.title('Agent\'s score')
    plt.xlabel('Episode')
    plt.ylabel('Score')


def plot_agent_stats(df, stats=None):
    sample = df['agent'][0]
    if stats:
        keys = [k for k in sample.keys() if k in stats]
    else:
        keys = sample.keys()
    fig, ax = plt.subplots(1, len(keys), figsize=(14, 4))
    for k, axis in zip(keys, ax):
        series_k = df['agent'].transform(lambda d: d[k])
        series_k.plot(ax=axis)
        axis.set_title(k)
        axis.set_xlabel('Episode')
    plt.tight_layout()
