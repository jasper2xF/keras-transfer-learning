import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_losses_path, val_losses_path):
    """Plot input losses with matplotlib."""
    train_losses = np.load(train_losses_path)
    val_losses = np.load(val_losses_path)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.savefig('loss.png')
    return None


def main():
    train_losses_path = sys.argv[1]
    val_losses_path = sys.argv[2]
    plot_losses(train_losses_path, val_losses_path)
    return None


if __name__ == "__main__":
    main()