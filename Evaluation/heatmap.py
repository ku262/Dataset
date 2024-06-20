from scipy.stats import gaussian_kde
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

def generate_heatmap(points, width, height):
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # Calculate kernel density estimation
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[0:width:complex(width), 0:height:complex(height)]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    heatmap = zi.reshape(xi.shape)

    # Normalize to 0-255
    min_value = np.min(heatmap)
    max_value = np.max(heatmap)
    heatmap_normalized = 255 * (heatmap - min_value) / (max_value - min_value)
    heatmap_normalized = np.uint8(np.clip(heatmap_normalized, 0, 255))

    return heatmap_normalized

def heat_map(point_contrix, last_str):
    # 1920x1080
    width, height = 1920, 1080
    # print(f"point num: {point_contrix.shape[0]}")

    random_indices = np.random.choice(point_contrix.shape[0], size=100, replace=False)
    heatmap = generate_heatmap(point_contrix[random_indices, 1:], width, height)

    plt.figure(figsize=(6.4, 3.6), dpi=300)
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(heatmap.T, cmap="turbo", origin='lower', alpha=1)

    plt.savefig(rf'.\imgs\heatmap_{last_str}.png', dpi=300)
    plt.show()

    image = cv2.imread(r'.\imgs\raw.png')
    image2 = cv2.imread(rf'.\imgs\heatmap_{last_str}.png')

    result = cv2.addWeighted(image, 1.0, image2, 0.7, 0)

    cv2.imwrite(rf'.\imgs\result_{last_str}.png', result)