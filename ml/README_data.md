# Dataset

## Source

UCI Bike Sharing, hour.csv, by Fanaee-T and Gama.
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## Versions

1. v1: January-June 2011 training data.

   Commit: 96a169ec67cc921392ff1b02ea1d89a02095e3bc
   DVC: 6dca5337b2a44cd400f411145c3c3118.dir

2. v2: Adds July-December 2011 to the training data.

   Commit: fc2c759f31d7ebd5c6fe0c9333050287549f8dc9
   DVC: 531a810a27543ccf6e981ecefd3ccc3f.dir

Both versions use the same January-March 2012 validation data.

## Storage

Data is tracked with DVC and stored in Azure. Git stores only data pointers.
