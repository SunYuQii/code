# Code

This is the implementation of the submission.

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:

  -  GPU:  NVIDIA GeForce RTX 3090.

  - CUDA: 12.0.
  - Driver version: 525.105.17
  - CPU: AMD EPYC 7542 32-core processor

- Software:

  - Python:  3.10.14
  - Pytorch:  2.0.1+cu11.8

You can pip install the `environment.yml` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset according to the following steps:

1. The raw dataset downloaded from website should be put into `/data/<fashion/book/yelp>/raw/`. The fashion and book datasets can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).The Yelp dataset can be obtained from [https://www.yelp.com/dataset](https://www.yelp.com/dataset). 
2. Conduct the preprocessing code `preData/1data_process.py` to filter cold-start users and items. After the procedure, you will get the id file  `/data/<fashion/book/yelp>/handled/id_map.json` and the interaction file  `/data/<fashion/book/yelp>/handled/inter_seq.txt`.
3. Convert the interaction file to the format used in this repo by running `data/2convert_inter.py`.

## Stage 1: Semantic Interaction Augmentor (SIA)

By SIA, you can get the augmented dataset.

1. Reverse pre-training SRS model.

   ```bash
   bash experiments/SASRec/SASRecFashionRev.bash
   ```

2. Run the CCG bash, and the result will be save to `/data/<fashion/book/yelp>/pseudo/`.

   ```bash
   bash experiments/CCG.bash
   ```

3. Run the SNF bash, and the derived dataset will be save to `/data/<fashion/book/yelp>/handled/interAug.txt`.

   ```bash
   bash experiments/SNF.bash
   ```

## Stage 2: Adaptive Reliability Validation (ARV)

By ARV, uou can get the reliability of each enhancement sequence.

1. Forward pre-training SRS model.

   ```bash
   bash experiments/SASRec/SASRecFashion.bash
   ```

2. Run the RCS bash, and the result will be save to `/data/<fashion/book/yelp>/reliability/`.

   ```bash
   bash experiments/RCS.bash
   ```

3. Run the LLM reason bash, and the result will be save to `/data/<fashion/book/yelp>/handled/weight.txt`.

   ```bash
   bash experiments/reason.bash
   ```

## Stage 3: Dual-Channel Training (DCT)

By DCT, you can get the final SRS models.

You can reproduce all LLMSeR experiments by running the bash as follows:

```bash
bash experiments/SASRec/AugSASRecFashion.bash
```

The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.