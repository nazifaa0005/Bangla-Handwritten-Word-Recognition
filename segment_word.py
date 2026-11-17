import argparse
from pathlib import Path

import cv2
import numpy as np


def fit_headline(ink):
    """Robust line through the TOP of the ink in each column.

    ink: 2-D array, 1 = ink. Returns (slope, intercept, support) where support is
    the fraction of ink columns that lie close to the line (about 1.0 means a clean
    headline, about 0 means there is no headline).
    """
    cols = np.where(ink.any(axis=0))[0]
    tops = ink[:, cols].argmax(axis=0)          # first ink row from the top, per column
    keep = np.ones(len(cols), bool)
    for _ in range(5):                          # fit, throw away outliers, fit again
        a, b = np.polyfit(cols[keep], tops[keep], 1)
        resid = np.abs(tops - (a * cols + b))
        tol = max(3.0, 2 * np.median(resid[keep]))
        keep = resid <= tol
    return a, b, keep.mean()


def level_word(gray, thresh=127, min_support=0.5):
    """Crop to the ink and rotate so the headline is horizontal."""
    ink = (gray > thresh).astype(np.uint8)
    ys, xs = np.where(ink > 0)
    g = np.pad(gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 25)
    a, b, support = fit_headline((g > thresh).astype(np.uint8))
    angle = 0.0
    if support >= min_support:                  # only rotate if a headline really exists
        angle = float(np.degrees(np.arctan(a)))
        h, w = g.shape
        rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        g = cv2.warpAffine(g, rotation, (w, h), flags=cv2.INTER_LINEAR)
    return g, angle


def segment_word(gray, thresh=127, kernel_frac=0.3, min_h_frac=0.40):
    """gray: 2-D uint8 image, WHITE strokes on BLACK. Returns (crops, info).

    kernel_frac : headline detector length as a fraction of the word's width
    min_h_frac  : ignore blobs shorter than this fraction of the word's height
                  (0.4 starts cutting off small marks; 0.15-0.3 all worked on kolom)
    """
    g, angle = level_word(gray, thresh)                         # step 1
    binary = (g > thresh // 2).astype(np.uint8) * 255
    ys, xs = np.where(binary > 0)
    ink_w, ink_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1

    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    stroke = max(2.0, 2 * np.percentile(dist[binary > 0], 90))  # rough stroke width

    # step 2a: long horizontal strokes
    k = max(15, int(kernel_frac * ink_w))
    erase = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    erase = cv2.dilate(erase, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(round(stroke)) + 4)))

    # step 2b: thin band around the fitted headline line
    a, b, support = fit_headline((binary > 0).astype(np.uint8))
    half = int(round(stroke / 2)) + 2
    has_line = support >= 0.45
    if has_line:
        for x in range(binary.shape[1]):
            y = a * x + b
            erase[int(max(0, y - half)):int(min(binary.shape[0], y + half + 1)), x] = 255
    no_headline = cv2.subtract(binary, erase)
    
    # step 3: blobs -> keep real letters
    count, _, stats, _ = cv2.connectedComponentsWithStats(no_headline, connectivity=8)
    spans = []
    for x, y, w, h, area in stats[1:]:
        centre_line = a * (x + w / 2) + b
        on_headline = has_line and y >= centre_line - half - 4 and y + h <= centre_line + half + 4
        if area >= max(10, 3 * stroke) and h >= min_h_frac * ink_h and not on_headline:
            spans.append([x, x + w])
    spans.sort()
    merged = []
    for span in spans:                          # merge blobs that overlap sideways
        if merged and span[0] < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span[1])
        else:
            merged.append(list(span))

    # step 4: full-height column crops from the levelled ORIGINAL
    crops = [g[:, max(0, x0 - 2):x1 + 2] for x0, x1 in merged]
    return crops, {"angle": angle, "spans": merged, "levelled": g, "stroke": stroke}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--white-on-black", action="store_true",
                        help="input already has white strokes on a black background")
    args = parser.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read {Path(args.image).resolve()}")
    if not args.white_on_black:
        gray = 255 - gray                       # dark-on-light drawing -> white-on-black

    crops, info = segment_word(gray)
    print(f"Headline angle corrected: {info['angle']:.1f} degrees")
    print(f"Estimated stroke width:   {info['stroke']:.1f} px")
    print(f"Letters found:            {len(crops)}")

    out = Path("segmented_characters")
    out.mkdir(exist_ok=True)
    preview = 255 - cv2.cvtColor(info["levelled"], cv2.COLOR_GRAY2BGR)
    for number, ((x0, x1), crop) in enumerate(zip(info["spans"], crops), 1):
        cv2.imwrite(str(out / f"word_char_{number}.png"), crop)
        cv2.rectangle(preview, (x0, 0), (x1, preview.shape[0] - 1), (0, 0, 255), 1)
        print(f"  letter {number}: x={x0}..{x1}")
    cv2.imwrite("word_regions_preview.png", preview)
    print("Saved crops to segmented_characters/ and word_regions_preview.png")


if __name__ == "__main__":
    main()
