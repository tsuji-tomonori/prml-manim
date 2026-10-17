"""Independent calculations for the two experiments and displayed numbers."""
import json
from pathlib import Path
import numpy as np

from kmeans_model import POINTS, INITIAL, ALTERNATE, run, toy_rgb_image, quantize

root = Path(__file__).resolve().parent
main, alternate = run(), run(initial=ALTERNATE)


def independent_cost(points, centres, labels):
    return sum(sum((float(x[i]) - float(centres[int(k), i])) ** 2
                   for i in range(points.shape[1]))
               for x, k in zip(points, labels))


assert [s["step"] for s in main] == ["E", "M", "E", "M", "E"]
assert [s["step"] for s in alternate] == ["E", "M", "E"]
assert [np.bincount(s["labels"], minlength=2).tolist() for s in main] == [
    [4, 10], [4, 10], [6, 8], [6, 8], [6, 8]]
assert np.allclose([s["J"] for s in main], [91.4, 28.588, 15.4196,
                                              11.264166666666666, 11.264166666666666], atol=1e-10)
assert abs(alternate[-1]["J"] - 10.56) < 1e-10
assert all(a["J"] + 1e-9 >= b["J"] for a, b in zip(main, main[1:]))
assert not np.array_equal(main[-1]["labels"], alternate[-1]["labels"])
for state in main + alternate:
    assert abs(independent_cost(POINTS, state["centres"], state["labels"]) - state["J"]) < 1e-9
    d0 = ((POINTS - state["centres"][0]) ** 2).sum(axis=1)
    d1 = ((POINTS - state["centres"][1]) ** 2).sum(axis=1)
    if state["step"] == "E":
        assert np.array_equal(state["labels"], np.where(d0 <= d1, 0, 1))
    else:
        for k in (0, 1):
            assert np.allclose(state["centres"][k], POINTS[state["labels"] == k].sum(axis=0) /
                               np.count_nonzero(state["labels"] == k))

image = toy_rgb_image()
assert image.shape == (12, 16, 3) and image.dtype == np.uint8
distortions = {}
for k in (2, 4):
    rebuilt, centres, labels, cost = quantize(image, k)
    assert len(np.unique(labels)) == k
    independently = np.sum((image.astype(float) - rebuilt.astype(float)) ** 2)
    assert abs(independently - cost) < 100  # RGB rounding changes at most 0.5 per channel.
    distortions[str(k)] = cost
assert distortions["4"] < distortions["2"]
bits = {"original": 192 * 24, "2": 24 * 2 + 192, "4": 24 * 4 + 192 * 2}
assert bits == {"original": 4608, "2": 240, "4": 480}
result = {
    "main": [{"step": s["step"], "J": s["J"], "counts": np.bincount(s["labels"], minlength=2).tolist(),
              "centres": s["centres"].tolist()} for s in main],
    "alternate_J": alternate[-1]["J"],
    "image_distortion": distortions,
    "theoretical_bits": bits,
    "checks": "passed",
}
(root / "numerical_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("PASS", [round(s["J"], 6) for s in main], round(alternate[-1]["J"], 6), bits)
