We report the results based on the released models in [GoogleDrive](https://drive.google.com/drive/folders/1aL5UkhXgevyoQDo_cLmmd-DUfZcAFRXu?usp=share_link), as in ```eval.log```.

Note that the results differ to that in ```train.log``` slightly, this is because of the BN statistics.
During training, the models are evaluated using 8 GPUs, in which each GPU has its own BN statistics, 
and we only save one model in GPU-0 as checkpoint, i.e., our released model.


| RepGhostNet | Params(M) | FLOPs(M) | Latency(ms) | Top-1 Acc.(%) | Top-5 Acc.(%) | checkpoints                                                                                          | logs                                                  |
|:------------|:----------|:---------|:------------|:--------------|:--------------|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------------|
| 0.5x        | 2.3       | 43       | 25.1        | 66.9          | 86.9          | [googledrive](https://drive.google.com/file/d/16AGg-kSscFXDpXPZ3cJpYwqeZbUlUoyr/view?usp=share_link) | [log](./train/repghostnet_0_5x_43M_66.95/train.log)   |
| 0.58x       | 2.5       | 60       | 31.9        | 68.9          | 88.4          | [googledrive](https://drive.google.com/file/d/1L6ccPjfnCMt5YK-pNFDfqGYvJyTRyZPR/view?usp=share_link) | [log](./train/repghostnet_0_58x_60M_68.94/train.log)  |
| 0.8x        | 3.3       | 96       | 44.5        | 72.2          | 90.5          | [googledrive](https://drive.google.com/file/d/13gmUpwiJF_O05f3-3UeEyKD57veL5cG-/view?usp=share_link) | [log](./train/repghostnet_0_8x_96M_72.24/train.log)   |
| 1.0x        | 4.1       | 142      | 62.2        | 74.2          | 91.5          | [googledrive](https://drive.google.com/file/d/1gzfGln60urfY38elpPHVTyv9b94ukn5o/view?usp=share_link) | [log](./train/repghostnet_1_0x_142M_74.22/train.log)  |
| 1.11x       | 4.5       | 170      | 71.5        | 75.1          | 92.2          | [googledrive](https://drive.google.com/file/d/14Lk4pKWIUFk1Mb53ooy_GsZbhMmz3iVE/view?usp=share_link) | [log](./train/repghostnet_1_11x_170M_75.07/train.log) |
| 1.3x        | 5.5       | 231      | 92.9        | 76.4          | 92.9          | [googledrive](https://drive.google.com/file/d/1dNHpX2JyiuTcDmmyvr8gnAI9t8RM-Nui/view?usp=share_link) | [log](./train/repghostnet_1_3x_231M_76.37/train.log)  |
| 1.5x        | 6.6       | 301      | 116.9       | 77.5          | 93.5          | [googledrive](https://drive.google.com/file/d/1TWAY654Dz8zcwhDBDN6QDWhV7as30P8e/view?usp=share_link) | [log](./train/repghostnet_1_5x_301M_77.45/train.log)  |
| 2.0x        | 9.8       | 516      | 190.0       | 78.8          | 94.3          | [googledrive](https://drive.google.com/file/d/12k00eWCXhKxx_fq3ewDhCNX08ftJ-iyP/view?usp=share_link) | [log](./train/repghostnet_2_0x_516M_78.81/train.log)  |
