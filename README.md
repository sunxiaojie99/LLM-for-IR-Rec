# LLM-for-IR-Rec

- [LLM-for-IR-Rec](#llm-for-ir-rec)
  - [LLM with IR](#llm-with-ir)
    - [LLM for IR](#llm-for-ir)
      - [For Pre-training LLM](#for-pre-training-llm)
      - [For Fine-tuning LLM](#for-fine-tuning-llm)
      - [For inference of LLM](#for-inference-of-llm)
      - [Joint Optimization of IR and LLM](#joint-optimization-of-ir-and-llm)
    - [IR for LLM](#ir-for-llm)
      - [Generating Synthetic Queries](#generating-synthetic-queries)
      - [Generating Synthetic Documents](#generating-synthetic-documents)
      - [Generating Ranking Lists](#generating-ranking-lists)
      - [Query Understanding](#query-understanding)
      - [Query Extension](#query-extension)
      - [Generate rather than Retrieve](#generate-rather-than-retrieve)
    - [Survey](#survey)
    - [Benchmark and Evaluation](#benchmark-and-evaluation)
  - [LLM with Rec](#llm-with-rec)
    - [Rec based on LLMs](#rec-based-on-llms)
    - [Rec enhanced by LLMs](#rec-enhanced-by-llms)
    - [Survey](#survey-1)
    - [other](#other)


## LLM with IR

### LLM for IR

#### For Pre-training LLM

- [REALM: Retrieval augmented language model pre-training.](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf) *Kelvin Guu et.al.* ICML 2020.(**REALM**)
- [Improving language models by retrieving from trillions of tokens.](https://arxiv.org/pdf/2112.04426.pdf) *Sebastian Borgeaud et.al.* ICML 2022. (**RETRO**)
- [Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study.](https://arxiv.org/pdf/2304.06762) *Boxin Wang et.al.* Arxiv 2023.



#### For Fine-tuning LLM

- [Dense Passage Retrieval for open-domain question answering.](https://arxiv.org/abs/2004.04906) *Vladimir Karpukhin et.al.* EMNLP 2020. (**DPR**)
- [RAG: Retrieval-augmented generation for knowledge-intensive NLP tasks.](https://arxiv.org/pdf/2005.12989) *Patrick Lewis et.al.* NeurIPS 2020. (**RAG**)
- [FiD: Leveraging passage retrieval with generative models for open domain question answering.](https://arxiv.org/pdf/2007.01282) *Gautier Izacard, Edouard Grave* EACL 2021.
- [Distilling Knowledge from Reader to Retriever for Question Answering.](https://openreview.net/forum?id=NTEz-6wysdb) Gautier Izacard et.al. ICLR 2021 (**FiD-KD**)
- [Copy Is All You Need.](https://arxiv.org/abs/2307.06962) *Tian Lan et.al.* ICLR 2023. (**COG**)



#### For inference of LLM

- [Generalization through memorization: Nearest neighbor language models.](https://arxiv.org/pdf/1911.00172.pdf) *Urvashi Khandelwal et.al.* Arxiv 2019.
- [Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.](https://arxiv.org/pdf/2212.10509) *Harsh Trivedi et.al.* Arxiv 2022.
- [Rethinking with retrieval: Faithful large language model inference.](https://arxiv.org/pdf/2301.00303) *Hangfeng He et.al.* Arxiv 2023.
- [Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation](https://arxiv.org/abs/2307.11019) *Ruiyang Ren et.al.* Arxiv 2023.



#### Joint Optimization of IR and LLM

- [Atlas: Few-shot Learning with Retrieval Augmented Language Models.](https://arxiv.org/pdf/2208.03299.pdf?trk=public_post_comment-text) *Gautier Izacard et.al.* Arxiv 2022.
- [REPLUG: Retrieval-Augmented Black-Box Language Models.](https://arxiv.org/pdf/2301.12652) *Weijia Shi et.al.* Arxiv 2023.(**REPLUG**)
- [Learning to Retrieve In-Context Examples for Large Language Models.](https://arxiv.org/pdf/2307.07164.pdf) *Liang Wang et.al.* Arxiv 2023.



### IR for LLM

#### Generating Synthetic Queries

- [InPars: Data augmentation for information retrieval using large language models.](https://arxiv.org/pdf/2202.05144) *Luiz Bonifacio et.al.* SIGIR 2022.
- [UPR: Improving passage retrieval with zero-shot question generation.](https://arxiv.org/pdf/2204.07496) *Devendra Singh Sachan et.al.* EMNLP 2022.
- [Promptagator: Fewshot dense retrieval from 8 examples.](https://arxiv.org/pdf/2209.11755) *Zhuyun Dai et.al.* ICLR 2023.

#### Generating Synthetic Documents

- [Precise Zero-Shot Dense Retrieval without Relevance Labels.](https://arxiv.org/pdf/2212.10496) *Luyu Gao et.al.* Arxiv 2022.
- [Generating Synthetic Documents for Cross-Encoder Re-Rankers: A Comparative Study of ChatGPT and Human Experts.](https://arxiv.org/pdf/2305.02320) *Arian Askari et.al.* Arxiv 2023.

#### Generating Ranking Lists

- [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent.](https://arxiv.org/pdf/2304.09542) *Weiwei Sun et.al.* Arxiv 2023.
- [Zero-Shot Listwise Document Reranking with a Large Language Model.](https://arxiv.org/pdf/2305.02156) *Xueguang Ma et.al.* Arxiv 2023.

#### Query Understanding

- [Query Understanding in the Age of Large Language Models.](https://arxiv.org/pdf/2306.16004) *Avishek Anand et.al.* Gen-IR 2023.

#### Query Extension

- [Generative relevance feedback with large language models.](https://arxiv.org/pdf/2304.13157) *Iain Mackie et.al.* Arxiv 2023.
- [Query2doc: Query expansion with large language models.](https://arxiv.org/pdf/2303.07678) *Liang Wang et.al.* Arxiv 2023.

#### Generate rather than Retrieve

- [Generate rather than retrieve: Large language models are strong context generators.](https://arxiv.org/pdf/2209.10063) *Wenhao Yu et.al.* ICLR 2023.

### Survey

**LLM for IR**

- [Perspectives on Large Language Models for Relevance Judgment](https://arxiv.org/pdf/2304.09161.pdf) *Guglielmo Faggioli et.al.* ICTIR 2023. (**Best paper**)
- [Large Language Models for Information Retrieval: A Survey](https://arxiv.org/pdf/2308.07107.pdf) *Yutao Zhu et.al.* Arxiv 2023.

**Retrieval Augmented LLM**

- [Retrieval-based Language Models and Applications](https://acl2023-retrieval-lm.github.io/) *Akari Asai et.al.* ACL 2023. (**Tutorial**)
- [Augmented Language Models: a Survey](https://arxiv.org/pdf/2302.07842.pdf) *Grégoire Mialon et.al.* Arxiv 2023.
- [Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community](https://arxiv.org/pdf/2307.09751.pdf) *Qingyao Ai et.al.* Arxiv 2023.

### Benchmark and Evaluation

- [KILT: a benchmark for knowledge intensive language tasks.](https://arxiv.org/pdf/2009.02252) *Fabio Petroni et.al.* NAACL 2021.



## LLM with Rec

### Rec based on LLMs

- [Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/abs/2203.13366). *Shijie Geng et.al.* RecSys  2022.

- [M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems.](https://arxiv.org/abs/2205.08084) Zeyu Cui et.al. arXiv 2022.

- [Prompt Distillation for Efficient LLM-based Recommendation](https://dl.acm.org/doi/10.1145/3583780.3615017) . *Lei Li et.al.* *CIKM 2023*

- [Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.](https://arxiv.org/abs/2305.07001)  Junjie Zhang  et.al. arXiv 2023.

- [Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.](https://arxiv.org/abs/2303.14524) Yunfan Gao et.al. arXiv 2023.

- [*TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.*](https://dl.acm.org/doi/fullHtml/10.1145/3604915.3608857) *Keqin Bao et.al.* *RecSys 2023.*

  

### Rec enhanced by LLMs



- [A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems.](https://arxiv.org/abs/2308.08434) Keqin Bao et.al. arXiv 2023.
- [CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation.](https://arxiv.org/abs/2310.19488) Yang Zhang et.al. arXiv 2023.
- [Large Language Model Can Interpret Latent Space of Sequential Recommender.](https://arxiv.org/abs/2310.20487) Zhengyi Yang et.al. arXiv 2023.

### Survey

- [Large Language Models for Generative Recommendation: A Survey and Visionary Discussions.](https://arxiv.org/abs/2309.01157) Lei Li et.al. arXiv 2023.

- [When Large Language Models Meet Personalization: Perspectives of Challenges and Opportunities.](https://arxiv.org/abs/2307.16376) Jin Chen et.al. arXiv 2023.

- [How Can Recommender Systems Benefit from Large Language Models: A Survey.](https://arxiv.org/abs/2306.05817) Jianghao Lin et.al. arXiv 2023.

- [Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems.](https://arxiv.org/pdf/2302.03735.pdf) Peng Liu et.al. arXiv 2023.

- [A Survey on Multi-Behavior Sequential Recommendation.](https://arxiv.org/abs/2308.15701) Xiaoqing Chen et.al. arXiv 2023.

- [Robust Recommender System: A Survey and Future Directions.](https://arxiv.org/pdf/2309.02057.pdf) Kaike Zhang et.al. arXiv 2023.

  

### other

- [*Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.*](https://dl.acm.org/doi/10.1145/3604915.3608860) *Jizhi Zhang et.al.* *RecSys 2023* 