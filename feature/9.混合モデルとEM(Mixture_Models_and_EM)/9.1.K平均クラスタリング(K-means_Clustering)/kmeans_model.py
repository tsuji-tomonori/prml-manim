"""Deterministic, inspectable examples for PRML 9.1."""
import numpy as np

POINTS = np.array([
    [-2.6, -1.5], [-2.3, -1.2], [-2.0, -1.7], [-1.7, -1.1],
    [-2.5, -0.6], [-1.6, -0.4], [-0.4, 0.0], [0.4, 0.3],
    [1.2, 1.1], [1.6, 0.8], [2.1, 1.4], [2.4, 0.8],
    [1.7, 1.9], [0.9, 1.8],
], dtype=float)
INITIAL = np.array([[-2.6, -1.5], [-1.6, -0.4]], dtype=float)
ALTERNATE = np.array([[-2.6, -1.5], [2.4, 0.8]], dtype=float)


def distortion(points, centres, labels):
    return float(np.sum((points - centres[labels]) ** 2))


def assign(points, centres):
    # argmin breaks exact ties toward the lower cluster number.
    return np.argmin(np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2), axis=1)


def recenter(points, labels, k):
    if any(not np.any(labels == j) for j in range(k)):
        raise ValueError("empty cluster: this example requires nonempty groups")
    return np.array([points[labels == j].mean(axis=0) for j in range(k)])


def run(points=POINTS, initial=INITIAL):
    centres = np.asarray(initial, dtype=float).copy()
    history = []
    previous = None
    for cycle in range(20):
        labels = assign(points, centres)
        history.append(dict(step="E", cycle=cycle, centres=centres.copy(),
                            labels=labels.copy(), J=distortion(points, centres, labels)))
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        centres = recenter(points, labels, len(centres))
        history.append(dict(step="M", cycle=cycle, centres=centres.copy(),
                            labels=labels.copy(), J=distortion(points, centres, labels)))
    else:
        raise RuntimeError("example did not settle")
    return history


def toy_rgb_image():
    """A 16×12 RGB drawing; each pixel is an independent K-means vector."""
    height, width = 12, 16
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if (x - 12) ** 2 + (y - 3) ** 2 <= 5:
                base = np.array([240, 205, 70])
            elif y >= 8:
                base = np.array([60, 155, 90])
            elif 4 <= x <= 6 and 5 <= y <= 9:
                base = np.array([205, 70, 75])
            else:
                base = np.array([72, 154, 225])
            variation = ((3*x + 5*y) % 13) - 6
            image[y, x] = np.clip(base + variation, 0, 255)
    return image


def quantize(image, k):
    pixels = image.reshape(-1, 3).astype(float)
    # Explicit seed pixels in the four color regions of the drawing.
    positions = [(0, 0), (2, 9), (5, 6), (12, 3)][:k]
    centres = np.array([image[y, x] for x, y in positions], dtype=float)
    for _ in range(30):
        labels = assign(pixels, centres)
        new = recenter(pixels, labels, k)
        if np.array_equal(new, centres):
            break
        centres = new
    labels = assign(pixels, centres)
    rebuilt = np.clip(np.rint(centres[labels]), 0, 255).astype(np.uint8).reshape(image.shape)
    return rebuilt, centres, labels, distortion(pixels, centres, labels)


if __name__ == "__main__":
    for state in run():
        print(state["step"], state["cycle"], round(state["J"], 6),
              np.bincount(state["labels"], minlength=2).tolist(),
              state["centres"].round(6).tolist())
