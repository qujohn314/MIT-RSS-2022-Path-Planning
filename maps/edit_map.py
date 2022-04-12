#!/usr/bin/env python

import numpy as np
import sys
import os
from PIL import Image
import argparse
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.morphology import erosion


def main():
    parser = argparse.ArgumentParser(description="Get some clicks")
    parser.add_argument("input_path", help="Path to input image file")

    args = parser.parse_args()
    input_file = args.input_path

    if not os.path.isfile(input_file):
        print("Error, need an input file.")
        sys.exit(1)
    
    footprint = disk(10)
    image = np.array(Image.open(input_file))
    image = np.mean(image, axis=2)
    new_image = erosion(image, footprint)

    output_filename = "eroded_map.jpeg"
    im = Image.fromarray(new_image).convert('RGB')
    im.save(output_filename)
    sys.exit(0)

if __name__ == "__main__":
    main()