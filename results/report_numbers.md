# Numbers used in Report V1 (rebuilt by scripts/report/make_figures.py)

Levels in %, with 95% item-bootstrap intervals in brackets (10,000 resamples of test items, seed 182);
differences in percentage points from a paired bootstrap over the same items. IFEval has no interval.

| model | dropped | obeyed | safe proc. | safe proc. (no counting) | dropped (no counting) | clean acc. | clean acc. (counting) | SEP attacked | transfer attacked | IFEval (mean of 4) |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained | 17.1 [15.0, 19.3] | 11.2 [9.4, 13.0] | 63.5 [60.7, 66.3] | 65.2 [61.9, 68.3] | 16.4 [14.1, 18.8] | 93.8 [92.3, 95.1] | 87.9 [84.0, 91.5] | 92.0 [90.8, 93.1] | 39.6 [37.5, 41.8] | 72.9 |
| ops-drop | 27.8 [25.3, 30.3] | 4.7 [3.5, 6.0] | 62.5 [59.7, 65.3] | 63.9 [60.6, 67.1] | 26.2 [23.3, 29.3] | 94.3 [93.0, 95.6] | 89.3 [85.7, 92.5] | 72.6 [70.6, 74.5] | 33.5 [31.4, 35.5] | 66.8 |
| ops-process | 5.5 [4.2, 6.8] | 6.5 [5.1, 8.0] | 87.3 [85.4, 89.2] | 87.0 [84.7, 89.2] | 6.6 [5.0, 8.4] | 82.4 [80.1, 84.5] | 38.4 [32.9, 44.0] | 83.0 [81.3, 84.6] | 36.9 [34.8, 39.1] | 71.1 |
| standard | 16.6 [14.5, 18.8] | 5.1 [3.9, 6.3] | 73.7 [71.2, 76.2] | 76.2 [73.3, 79.0] | 14.1 [11.8, 16.4] | 89.5 [87.7, 91.2] | 77.5 [73.0, 82.1] | 28.1 [26.2, 30.0] | 11.2 [9.8, 12.6] | 69.0 |
| standard+fid | 5.0 [3.8, 6.2] | 4.5 [3.3, 5.7] | 88.7 [86.9, 90.5] | 88.0 [85.8, 90.2] | 5.5 [3.9, 7.0] | 81.7 [79.5, 83.9] | 40.4 [34.9, 45.9] | 19.9 [18.2, 21.6] | 6.3 [5.3, 7.4] | 68.5 |
| ops-drop SFT | 28.1 [25.5, 30.7] | 5.1 [3.9, 6.4] | 62.5 [59.7, 65.2] | 64.8 [61.6, 67.9] | 25.6 [22.6, 28.6] | 93.1 [91.6, 94.5] | 88.3 [84.4, 91.9] | 75.2 [73.4, 77.1] | 34.8 [32.8, 37.0] |  |
| ops-process SFT | 7.4 [5.9, 8.9] | 5.0 [3.8, 6.2] | 85.5 [83.5, 87.5] | 85.2 [82.8, 87.6] | 7.9 [6.2, 9.8] | 90.5 [88.8, 92.1] | 70.7 [65.5, 75.6] | 76.7 [74.9, 78.5] | 32.4 [30.3, 34.5] |  |
| standard SFT | 23.2 [20.8, 25.6] | 4.4 [3.3, 5.6] | 66.6 [63.9, 69.3] | 67.6 [64.3, 70.7] | 22.0 [19.2, 24.7] | 90.7 [89.0, 92.3] | 84.7 [80.5, 88.6] | 22.1 [20.2, 23.9] |  |  |
| standard+fid SFT | 21.1 [18.8, 23.5] | 4.1 [3.0, 5.3] | 69.1 [66.4, 71.7] | 67.7 [64.5, 70.8] | 22.2 [19.5, 25.0] | 88.3 [86.4, 90.1] | 74.3 [69.4, 79.2] | 14.3 [12.8, 15.9] |  |  |
| half-and-half SFT | 16.4 [14.4, 18.6] | 4.5 [3.4, 5.8] | 74.9 [72.3, 77.4] | 75.4 [72.4, 78.3] | 15.7 [13.4, 18.1] | 90.4 [88.7, 92.0] | 76.5 [71.7, 81.1] | 29.1 [27.2, 31.2] |  |  |
| standard+process SFT | 19.4 [17.2, 21.7] | 3.9 [2.7, 5.1] | 71.9 [69.3, 74.4] | 71.0 [67.8, 74.0] | 19.9 [17.3, 22.6] | 90.2 [88.5, 91.9] | 79.5 [74.9, 83.7] | 15.4 [13.9, 17.0] |  |  |
| system-user | 6.8 [5.4, 8.3] | 7.4 [5.9, 8.9] | 85.1 [83.0, 87.1] | 84.8 [82.3, 87.1] | 7.4 [5.7, 9.2] | 83.7 [81.5, 85.8] | 44.6 [39.1, 50.2] | 85.8 [84.3, 87.3] | 38.8 [36.6, 40.9] | 72.1 |
| ops-process+fid | 2.6 [1.7, 3.5] | 1.4 [0.8, 2.1] | 80.9 [78.6, 83.1] | 94.1 [92.5, 95.6] | 2.7 [1.6, 3.8] | 77.5 [75.0, 79.9] | 18.2 [14.0, 22.8] | 46.6 [44.5, 48.8] | 13.1 [11.6, 14.5] | 66.6 |
| ops-process+fid SFT | 2.5 [1.6, 3.4] | 1.7 [1.0, 2.5] | 90.6 [88.9, 92.2] | 94.7 [93.1, 96.2] | 2.6 [1.5, 3.7] | 79.3 [76.9, 81.5] | 25.1 [20.2, 30.0] |  |  |  |
| half-and-half | 11.7 [9.9, 13.6] | 4.5 [3.3, 5.7] | 82.1 [79.9, 84.2] | 83.6 [81.1, 86.1] | 10.3 [8.4, 12.4] | 89.0 [87.2, 90.8] | 65.5 [60.3, 70.7] | 37.8 [35.7, 40.0] |  |  |
| half-and-half+fid | 3.6 [2.6, 4.7] | 4.9 [3.7, 6.2] | 90.1 [88.4, 91.8] | 89.1 [87.0, 91.2] | 4.3 [3.0, 5.7] | 78.4 [76.0, 80.7] | 25.1 [20.2, 30.0] | 28.6 [26.7, 30.6] |  |  |
| ops-process, no counting ops | 9.2 [7.5, 10.9] | 6.8 [5.4, 8.2] | 83.8 [81.8, 85.9] | 85.6 [83.2, 87.9] | 7.7 [5.9, 9.5] | 87.6 [85.6, 89.4] | 58.3 [52.8, 63.8] |  |  |  |
| ops-process+fid, no counting ops | 2.8 [1.9, 3.9] | 2.3 [1.5, 3.3] | 94.2 [92.8, 95.5] | 94.2 [92.6, 95.7] | 2.7 [1.6, 3.8] | 87.6 [85.6, 89.4] | 56.0 [50.5, 61.6] |  |  |  |
| standard 2x | 13.1 [11.2, 15.2] | 4.9 [3.7, 6.2] | 78.7 [76.4, 81.0] | 79.4 [76.7, 82.1] | 12.4 [10.3, 14.8] | 91.3 [89.6, 92.9] | 79.5 [74.9, 83.7] | 21.9 [20.1, 23.8] |  |  |
| standard 11x | 18.1 [15.9, 20.3] | 4.9 [3.7, 6.1] | 72.4 [69.9, 75.0] | 75.7 [72.8, 78.5] | 14.6 [12.3, 17.1] | 91.7 [90.1, 93.2] | 84.7 [80.5, 88.6] | 15.0 [13.5, 16.6] |  |  |
| Meta-SecAlign-8B | 27.5 [24.9, 30.1] | 3.1 [2.1, 4.1] | 62.5 [59.7, 65.2] | 54.6 [51.2, 58.0] | 33.1 [30.0, 36.4] | 92.3 [90.8, 93.8] | 88.3 [84.7, 91.9] | 10.3 [9.0, 11.7] | 3.9 [3.0, 4.8] | 80.4 |
| SecPO-8B | 32.7 [30.1, 35.4] | 4.1 [3.0, 5.3] | 60.1 [57.4, 62.9] | 50.1 [46.7, 53.4] | 41.7 [38.4, 45.1] | 90.3 [88.6, 92.0] | 74.6 [69.7, 79.5] | 7.4 [6.3, 8.6] | 2.5 [1.8, 3.1] | 70.8 |
| Llama-3.1-8B | 15.4 [13.4, 17.6] | 14.8 [12.8, 16.9] | 67.0 [64.3, 69.7] | 62.0 [58.7, 65.3] | 17.8 [15.3, 20.3] | 95.3 [94.1, 96.5] | 89.3 [85.7, 92.5] | 98.7 [98.1, 99.2] | 38.4 [36.3, 40.6] | 80.2 |

Held-out conflicts, share answered exactly as the processing target (%):

| model | user-input conflicts | system-user conflicts | no conflict (fidelity) |
|---|---|---|---|
| untrained | 33.3 | 27.0 | 81.0 |
| ops-drop | 3.7 | 1.0 | 50.3 |
| ops-process | 79.0 | 31.0 | 85.0 |
| standard | 62.7 | 51.0 | 88.0 |
| standard+fid | 66.0 | 63.7 | 65.3 |
| standard SFT | 61.0 | 51.7 | 89.0 |
| standard+fid SFT | 90.0 | 65.7 | 98.3 |
| half-and-half SFT | 89.7 | 60.0 | 98.3 |
| system-user | 53.7 | 66.3 | 56.0 |
| ops-process+fid | 63.0 | 36.3 | 69.7 |
| ops-process+fid SFT | 96.0 | 90.7 | 99.7 |
| half-and-half | 88.0 | 56.0 | 98.0 |
| half-and-half+fid | 83.7 | 67.0 | 88.3 |

| comparison | quantity | difference [95% CI] |
|---|---|---|
| ops-process - ops-drop | safe processing | 24.8 [22.1, 27.6] |
| ops-process - ops-drop | safe processing, no counting | 23.1 [20.0, 26.2] |
| ops-process - ops-drop | dropped | -22.3 [-24.9, -19.8] |
| ops-process - ops-drop | dropped, no counting | -19.6 [-22.6, -16.8] |
| ops-process - ops-drop | SEP attacked | 10.3 [8.6, 12.2] |
| ops-process - ops-drop | transfer attacked | 3.5 [1.4, 5.5] |
| ops-process SFT - ops-drop SFT | safe processing | 23.0 [20.4, 25.6] |
| ops-process SFT - ops-drop SFT | safe processing, no counting | 20.4 [17.5, 23.3] |
| ops-process SFT - ops-drop SFT | dropped | -20.7 [-23.3, -18.2] |
| ops-process SFT - ops-drop SFT | dropped, no counting | -17.7 [-20.6, -14.9] |
| ops-process SFT - ops-drop SFT | SEP attacked | 1.5 [-0.2, 3.1] |
| ops-process SFT - ops-drop SFT | transfer attacked | -2.5 [-4.5, -0.4] |
| ops-drop - standard | safe processing | -11.2 [-13.8, -8.7] |
| ops-drop - standard | safe processing, no counting | -12.3 [-15.1, -9.4] |
| ops-drop - standard | dropped | 11.2 [8.7, 13.7] |
| ops-drop - standard | dropped, no counting | 12.2 [9.5, 15.0] |
| ops-drop - standard | SEP attacked | 44.5 [42.1, 46.8] |
| ops-drop - standard | transfer attacked | 22.2 [20.2, 24.3] |
| ops-drop SFT - standard SFT | safe processing | -4.1 [-6.7, -1.5] |
| ops-drop SFT - standard SFT | safe processing, no counting | -2.8 [-5.6, 0.0] |
| ops-drop SFT - standard SFT | dropped | 4.9 [2.6, 7.3] |
| ops-drop SFT - standard SFT | dropped, no counting | 3.6 [1.0, 6.3] |
| ops-drop SFT - standard SFT | SEP attacked | 53.2 [50.9, 55.5] |
| ops-drop - untrained | safe processing | -1.0 [-3.8, 1.7] |
| ops-drop - untrained | safe processing, no counting | -1.3 [-4.4, 1.9] |
| ops-drop - untrained | dropped | 10.7 [8.0, 13.4] |
| ops-drop - untrained | dropped, no counting | 9.9 [7.0, 12.8] |
| ops-drop - untrained | SEP attacked | -19.4 [-21.2, -17.4] |
| ops-drop - untrained | transfer attacked | -6.2 [-8.6, -3.9] |
| standard SFT - untrained | safe processing | 3.1 [0.3, 5.8] |
| standard SFT - untrained | safe processing, no counting | 2.4 [-0.6, 5.5] |
| standard SFT - untrained | dropped | 6.1 [3.5, 8.6] |
| standard SFT - untrained | dropped, no counting | 5.6 [2.8, 8.4] |
| standard SFT - untrained | SEP attacked | -69.9 [-72.0, -67.8] |
| standard+fid SFT - standard SFT | safe processing | 2.5 [0.6, 4.5] |
| standard+fid SFT - standard SFT | safe processing, no counting | 0.1 [-2.0, 2.2] |
| standard+fid SFT - standard SFT | dropped | -2.1 [-3.9, -0.2] |
| standard+fid SFT - standard SFT | dropped, no counting | 0.2 [-1.7, 2.3] |
| standard+fid SFT - standard SFT | SEP attacked | -7.7 [-9.4, -6.1] |
| half-and-half SFT - standard SFT | safe processing | 8.3 [6.3, 10.3] |
| half-and-half SFT - standard SFT | safe processing, no counting | 7.8 [5.7, 9.9] |
| half-and-half SFT - standard SFT | dropped | -6.8 [-8.6, -4.9] |
| half-and-half SFT - standard SFT | dropped, no counting | -6.3 [-8.4, -4.3] |
| half-and-half SFT - standard SFT | SEP attacked | 7.1 [5.3, 8.9] |
| standard+process SFT - standard SFT | safe processing | 5.3 [3.4, 7.2] |
| standard+process SFT - standard SFT | safe processing, no counting | 3.4 [1.4, 5.5] |
| standard+process SFT - standard SFT | dropped | -3.8 [-5.5, -2.0] |
| standard+process SFT - standard SFT | dropped, no counting | -2.1 [-3.9, -0.2] |
| standard+process SFT - standard SFT | SEP attacked | -6.7 [-8.3, -5.1] |
| standard+process SFT - half-and-half SFT | safe processing | -3.0 [-4.7, -1.4] |
| standard+process SFT - half-and-half SFT | safe processing, no counting | -4.4 [-6.3, -2.7] |
| standard+process SFT - half-and-half SFT | dropped | 3.0 [1.5, 4.5] |
| standard+process SFT - half-and-half SFT | dropped, no counting | 4.2 [2.6, 5.9] |
| standard+process SFT - half-and-half SFT | SEP attacked | -13.8 [-15.7, -11.9] |
| (ops-process SFT - ops-drop SFT) - (ops-process - ops-drop) | SEP attacked | -8.9 [-11.1, -6.7] |
