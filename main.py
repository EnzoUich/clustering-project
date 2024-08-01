#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


def main() -> None:
    data = pd.read_csv("data/data.csv")

    # Split off old cluster
    original_cluster = data.pop("CLUSTER")
    original_cluster = original_cluster.map(
        {"Cluster 0": 0, "Cluster 1": 1, "Cluster 2": 2, "Cluster 3": 3}
    )

    # Remove label columns
    data.drop(data.columns[[0, 1]], axis=1, inplace=True)
    arr = np.array(data)

    # Cluster data
    kmeans = KMeans(n_clusters=4, random_state=6)
    kmeans.fit(arr)
    new_cluster = kmeans.labels_

    # Compute metrics
    accuracy = metrics.accuracy_score(original_cluster, new_cluster)
    print(f"Accuracy: {accuracy:.2%}")
    confusion_matrix = metrics.confusion_matrix(original_cluster, new_cluster)

    # Show confusion matrix
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[0, 1, 2, 3]
    )
    cm_display.plot()
    plt.show()


if __name__ == "__main__":
    main()
