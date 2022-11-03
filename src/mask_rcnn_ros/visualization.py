import colorsys
import random
import numpy as np
from skimage.measure import find_contours
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from mask_rcnn_ros.config import COCO_CLASS_NAMES


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(float(i) / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image


def generate_masked_image(image, masks, colors):
    masked_image = image.astype(np.uint32).copy()
    masks = masks.cpu().detach().numpy()

    for mask, color in zip(masks, colors):
        apply_mask(masked_image, mask[0], color)

    return masked_image


def add_bounding_box(ax, bbox, color, score, label):
    x1, y1, x2, y2 = bbox
    p = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        alpha=0.7,
        linestyle="dashed",
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(p)

    x = random.randint(int(x1), (x1 + x2) // 2)
    caption = "{} {:.3f}".format(label, score) if score else label
    ax.text(
        x1,
        y1 + 8,
        caption,
        color="black",
        size=17,
        bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black"),
    )


def add_padded_mask(ax, mask):

    padded_mask = np.zeros((mask.shape[1] + 2, mask.shape[2] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask[0]
    contours = find_contours(padded_mask, 0.5)

    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = patches.Polygon(verts, facecolor="none", edgecolor="black", lw=2)
        ax.add_patch(p)


def plot_result(image, result, ros_format=False):
    N = result["scores"].shape[0]

    fig, ax = plt.subplots(1, figsize=(16, 16))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis("off")

    masked_image = generate_masked_image(image, result["masks"], colors)

    for i in range(N):
        class_id = result["labels"][i].item()
        label = COCO_CLASS_NAMES[class_id]
        bbox = result["boxes"][i].cpu().detach().numpy()
        score = result["scores"][i].item()
        mask = result["masks"][i].cpu().detach().numpy()

        add_bounding_box(ax, bbox, colors[i], score, label)
        add_padded_mask(ax, mask)

    ax.imshow(masked_image.astype(np.uint8))

    if ros_format:
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        plt.close(fig)
        return img
