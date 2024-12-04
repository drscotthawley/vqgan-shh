from torchvision.utils import make_grid
import wandb
import matplotlib.pyplot as plt
import tempfile
import numpy as np


def viz_codebook(model, config, epoch):
    if config.no_wandb: return
    # Extract VQ codebook vectors
    codebook_vectors = model.vq.codebook.detach().cpu().numpy()
    
    # Reshape the codebook vectors to the desired shape
    codebook_image = codebook_vectors.reshape(config.vq_num_embeddings, config.vq_embedding_dim)
    
    # Create an image of the codebook vectors using matplotlib
    plt.figure(figsize=(16, 4))
    plt.imshow(codebook_image.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('VQ Codebook Vectors')
    plt.ylabel('Embedding Dimension')
    plt.xlabel('Codebook Index')
    
    # Adjust layout to remove extra margins and whitespace
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()
        
        # Log the image to wandb
        wandb.log({
            'codebook/image': wandb.Image(tmpfile.name, caption=f'Epoch {epoch} - VQ Codebook Vectors')
        })
    
    plt.close()
    
    # Compute the magnitudes of the codebook vectors
    magnitudes = np.linalg.norm(codebook_vectors, axis=1)

    # Create a figure with one row and two columns for the histograms
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    # Plot the histogram of magnitudes
    axs[0].hist(magnitudes, bins=50, color='blue', edgecolor='black')
    axs[0].set_title('Histogram of Codebook Vector Magnitudes')
    axs[0].set_xlabel('Magnitude')
    axs[0].set_ylabel('Frequency')

    # Plot the histogram of elements
    axs[1].hist(codebook_vectors.flatten(), bins=200, color='blue', edgecolor='black')
    axs[1].set_title('Histogram of Codebook Vector Elements')
    axs[1].set_xlabel('Element Value')
    axs[1].set_ylabel('Frequency')

    # Adjust layout to remove extra margins and whitespace
    plt.tight_layout()

    # Save the histogram image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()

        # Log the histogram image to wandb
        wandb.log({
            'codebook/histograms': wandb.Image(tmpfile.name, caption=f'Epoch {epoch} - Histograms of Codebook Vectors')
        })

    plt.close()

