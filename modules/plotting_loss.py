import numpy as np
import matplotlib.pyplot as plt

def plotingstuff(epoch,stars,dashes):
    x = np.linspace(0, epoch - 1, epoch)

    fig, ax1 = plt.subplots()
    ax1.plot(x, losses, label="loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f"AnomCLR Loss with {args.n_jets:.0e} jets")
    ax1.legend()
    
    plt.savefig((expt_dir+f"CLR-Loss_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")

    # Create a new figure and axes for the second plot
    fig, ax2 = plt.subplots()
    ax2.plot(x, stars, label="Anomaly Similarity")
    ax2.plot(x, dashes, label="Physical Similarity")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Similarity of the Transformer Network')
    ax2.legend()
    plt.savefig((expt_dir+f"Similarities_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")