# LLM-for-IR-Rec

> Note:
>
> - This project is based on [Awesome Information Retrieval in the Age of Large Language Model](https://github.com/IR-LLM/Awesome-Information-Retrieval-in-the-Age-of-Large-Language-Model#generating-synthetic-documents) and [LLM4IR-Survey](https://github.com/RUC-NLPIR/LLM4IR-Survey).
> - Any feedback and contribution are welcome, please open an issue or contact me.
>




- [LLM-for-IR-Rec](#llm-for-ir-rec)
  - [LLM with IR](#llm-with-ir)
    - [IR for LLM](#ir-for-llm)
      - [For Pre-training LLM](#for-pre-training-llm)
      - [For Fine-tuning LLM](#for-fine-tuning-llm)
      - [For inference of LLM](#for-inference-of-llm)
      - [Joint Optimization of IR and LLM](#joint-optimization-of-ir-and-llm)
    - [LLM for IR](#llm-for-ir)
      - [Retriever](#retriever)
        - [Generating Synthetic Queries](#generating-synthetic-queries)
        - [Generating Synthetic Documents](#generating-synthetic-documents)
        - [Employing LLMs to Enhance Model Architecture](#employing-llms-to-enhance-model-architecture)
        - [Using LLM as Embedder](#using-llm-as-embedder)
        - [Generate rather than Retrieve](#generate-rather-than-retrieve)
      - [Re-ranker](#re-ranker)
        - [Generating Synthetic Queries](#generating-synthetic-queries-1)
        - [Generating Synthetic Documents](#generating-synthetic-documents-1)
        - [Fine-tuning LLMs for Reranking](#fine-tuning-llms-for-reranking)
        - [Prompting LLMs for Reranking](#prompting-llms-for-reranking)
      - [Query Rewriter](#query-rewriter)
        - [Query Understanding](#query-understanding)
        - [Query Extension](#query-extension)
        - [Query Rewriting](#query-rewriting)
    - [Analysis paper](#analysis-paper)
    - [Survey](#survey)
    - [Benchmark and Evaluation](#benchmark-and-evaluation)
  - [LLM with Rec](#llm-with-rec)
    - [Rec based on LLMs](#rec-based-on-llms)
    - [Rec enhanced by LLMs](#rec-enhanced-by-llms)
    - [Survey](#survey-1)
    - [other](#other)




## LLM with IR

### IR for LLM

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
- [Active Retrieval Augmented Generation.](https://arxiv.org/abs/2305.06983)  Zhengbao Jiang et.al. Arxiv 2023. (**FLARE**)



#### For inference of LLM

- [Generalization through memorization: Nearest neighbor language models.](https://arxiv.org/pdf/1911.00172.pdf) *Urvashi Khandelwal et.al.* ﻿ICLR 2020. (**kNN-LM**)
- [Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.](https://aclanthology.org/2023.acl-long.557.pdf) *Harsh Trivedi et.al.* ACL 2023. (**IRCoT QA**)
- [Rethinking with retrieval: Faithful large language model inference.](https://arxiv.org/pdf/2301.00303) *Hangfeng He et.al.* Arxiv 2023.
- [Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation](https://arxiv.org/abs/2307.11019) *Ruiyang Ren et.al.* Arxiv 2023.
- [Retrieve Anything To Augment Large Language Models.](https://arxiv.org/abs/2310.07554) Peitian Zhang et.al. Arxiv 2023. (**LLM-Embedder**, supports the diverse retrieval augmentation needs of LLMs with one unified embedding model)


#### Joint Optimization of IR and LLM

- [Atlas: Few-shot Learning with Retrieval Augmented Language Models.](https://arxiv.org/pdf/2208.03299.pdf?trk=public_post_comment-text) *Gautier Izacard et.al.* Arxiv 2022.
- [REPLUG: Retrieval-Augmented Black-Box Language Models.](https://arxiv.org/pdf/2301.12652) *Weijia Shi et.al.* Arxiv 2023.(**REPLUG**)
- [Learning to Retrieve In-Context Examples for Large Language Models.](https://arxiv.org/pdf/2307.07164.pdf) *Liang Wang et.al.* Arxiv 2023.



### LLM for IR



#### Retriever

##### Generating Synthetic Queries

- [InPars: Data augmentation for information retrieval using large language models.](https://arxiv.org/pdf/2202.05144) *Luiz Bonifacio et.al.* SIGIR 2022. (**InPars**)
- [UPR: Improving passage retrieval with zero-shot question generation.](https://arxiv.org/pdf/2204.07496) *Devendra Singh Sachan et.al.* EMNLP 2022. (**UPR, point-wise, query generation**)
- [Promptagator: Fewshot dense retrieval from 8 examples.](https://arxiv.org/pdf/2209.11755) *Zhuyun Dai et.al.* ICLR 2023. (**Promptagator**)



##### Generating Synthetic Documents

- [Precise Zero-Shot Dense Retrieval without Relevance Labels.](https://arxiv.org/pdf/2212.10496) *Luyu Gao et.al.* Arxiv 2022. (**HyDE**)

  

##### Employing LLMs to Enhance Model Architecture

- [Text and Code Embeddings by Contrastive Pre-Training](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf), *Neelakantan et al.*, arXiv 2022.
- [Large Dual Encoders Are Generalizable Retrievers](https://aclanthology.org/2022.emnlp-main.669.pdf), *Ni et al.*, ACL 2022. 
- [Task-aware Retrieval with Instructions](https://aclanthology.org/2023.findings-acl.225.pdf), *Asai et al.*, ACL 2023 (Findings).
- [Transformer memory as a differentiable search index.](https://proceedings.neurips.cc/paper_files/paper/2022/file/892840a6123b5ec99ebaab8be1530fba-Paper-Conference.pdf) *Tay et al.*, NeurIPS 2022.


##### Using LLM as Embedder
- [Vector Search with OpenAI Embeddings: Lucene Is All You Need.](https://arxiv.org/abs/2308.14963) Jimmy Lin et al. arXiv 2023. (**OpenAI’s ada2 embedding+HNSW(Lucene search library), rather than Faiss library**)
- [Language Models are Universal Embedders.](https://arxiv.org/abs/2310.08232) Xin Zhang et.al. arXiv 2023. (**across tasks, natural and programming languages**)



##### Generate rather than Retrieve

- [Generate rather than retrieve: Large language models are strong context generators.](https://arxiv.org/pdf/2209.10063) *Wenhao Yu et.al.* ICLR 2023. (**Retrieve-then-Read->Generate-then-Read pipeline**)
- [Large Language Models are Built-in Autoregressive Search Engines](https://aclanthology.org/2023.findings-acl.167.pdf), *Ziems et al.*, ACL 2023 (Findings). (**LLM-URL, Given query, llm directly generate Web URLs, open domain QA**)



#### Re-ranker

##### Generating Synthetic Queries



##### Generating Synthetic Documents

- [Generating Synthetic Documents for Cross-Encoder Re-Rankers: A Comparative Study of ChatGPT and Human Experts.](https://arxiv.org/pdf/2305.02320) *Arian Askari et.al.* Arxiv 2023. (**ChatGPT-RetrievalQA dataset**)

  

##### Fine-tuning LLMs for Reranking

- [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://aclanthology.org/2020.findings-emnlp.63.pdf), *Nogueira et al.*, EMNLP 2020 (Findings). 
- [Text-to-Text Multi-view Learning for Passage Re-ranking](https://dl.acm.org/doi/pdf/10.1145/3404835.3463048), *Ju et al.*, SIGIR 2021 (Short Paper). 
- [The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models](https://arxiv.org/pdf/2101.05667.pdf), *Pradeep et al.*, arXiv 2021. 
- [RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses](https://dl.acm.org/doi/pdf/10.1145/3539618.3592047), *Zhuang et al.*, SIGIR 2023 (**fine-tune T5 with ranking losses**). 



##### Prompting LLMs for Reranking

- [Improving Passage Retrieval with Zero-Shot Question Generation](https://aclanthology.org/2022.emnlp-main.249.pdf), *Sachan et al.*, EMNLP 2022. 

- [Discrete Prompt Optimization via Constrained Generation for Zero-shot Re-ranker](https://aclanthology.org/2023.findings-acl.61.pdf), *Cho et al.*, ACL 2023 (Findings).

- [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent.](https://arxiv.org/pdf/2304.09542) *Weiwei Sun et.al.* Arxiv 2023. (**RankGPT, list-wise, sliding window**)

- [Zero-Shot Listwise Document Reranking with a Large Language Model.](https://arxiv.org/pdf/2305.02156) *Xueguang Ma et.al.* Arxiv 2023. (**LRL, list-wise, sliding window**)

- [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.](https://arxiv.org/pdf/2306.17563.pdf) *Qin et al.* arXiv 2023. (**pairwise**)
  
- [Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels.](https://arxiv.org/pdf/2310.14122.pdf) Honglei Zhuang et.al. arXiv 2023. 
  

#### Query Rewriter

##### Query Understanding

- [Query Understanding in the Age of Large Language Models.](https://arxiv.org/pdf/2306.16004) *Avishek Anand et.al.* Gen-IR 2023.

##### Query Extension

- [Generative relevance feedback with large language models.](https://arxiv.org/pdf/2304.13157) *Iain Mackie et.al.* Arxiv 2023.
- [Query2doc: Query expansion with large language models.](https://arxiv.org/pdf/2303.07678) *Liang Wang et.al.* Arxiv 2023.
- [Generate, Filter, and Fuse: Query Expansion via Multi-Step Keyword Generation for Zero-Shot Neural Rankers.](https://arxiv.org/pdf/2311.09175.pdf) Minghan Li et.al. arXiv 2023. (**GFF, a query expansion pipeline to imporve zero-shot ranker**)

##### Query Rewriting

- [Large Language Model based Long-tail Query Rewriting in Taobao Search.](https://arxiv.org/pdf/2311.03758.pdf) *Wenjun Peng et.al.* Arxiv 2023.




### Analysis paper

- [LLMs may Dominate Information Access: Neural Retrievers are Biased Towards LLM-Generated Texts.](https://arxiv.org/abs/2310.20501) Sunhao Dai et.al. Arxiv 2023.

  

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
  
- [LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking.](https://arxiv.org/pdf/2311.02089.pdf) Zhenrui Yue et.al. arXiv 2023.(**LLM ranker**, map the item to the corresponding index letter)

  

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

- [ Is ChatGPT a Good Recommender? A Preliminary Study.](https://arxiv.org/abs/2304.10149) Junling Liu et.al. CIKM 2023 GenRec Workshop.

  

### other

- [*Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.*](https://dl.acm.org/doi/10.1145/3604915.3608860) *Jizhi Zhang et.al.* *RecSys 2023* 