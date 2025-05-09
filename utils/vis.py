from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def tsne_vis(sentence_embeddings, word_labels):
    # reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sentence_embeddings)

    # plot the embeddings
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "orange", "purple", "green", "brown"]
    for i, label in enumerate(word_labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i], s=100)
        plt.text(embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1] + 0.1, label, fontsize=12, color=colors[i])
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE Visualization of Word Embeddings")
    plt.show()

def plot_attention(attn_matrix, tokens, title="Attention Heatmap"):
    plt.figure(figsize=(10, 8))  # Set the figure size
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)  # Plot the attention matrix as a heatmap
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title(title)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.yticks(rotation=0)  # Rotate y-axis labels
    plt.show()