
# Denoising Bottleneck with Mutual Information Maximization 
for Video Multimodal Fusion

###### Abstract

Video multimodal fusion aims to integrate multimodal signals in videos, such as visual, audio and text, to make a complementary prediction with multiple modalities contents. However, unlike other image-text multimodal tasks, video has longer multimodal sequences with more redundancy and noise in both visual and audio modalities. Prior denoising methods like forget gate are coarse in the granularity of noise filtering. They often suppress the redundant and noisy information at the risk of losing critical information. Therefore, we propose a denoising bottleneck fusion (DBF) model for fine-grained video multimodal fusion. On the one hand, we employ a bottleneck mechanism to filter out noise and redundancy with a restrained receptive field. On the other hand, we use a mutual information maximization module to regulate the filter-out module to preserve key information within different modalities. Our DBF model achieves significant improvement over current state-of-the-art baselines on multiple benchmarks covering multimodal sentiment analysis and multimodal summarization tasks. It proves that our model can effectively capture salient features from noisy and redundant video, audio, and text inputs. The code for this paper is publicly available at <https://github.com/WSXRHFG/DBF>.  

## 1 Introduction

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: 
An example of redundancy and noise in a video.
As illustrated, consecutive frames have high cosine similarity, which results in a problem of redundancy.
In addition, useless information like distracting background and weak alignment between frames and transcripts compose noises in videos.
[/FIGURE]

With the rapid development of social platforms and digital devices, more and more videos are flooding our lives, which leads video multimodal fusion an increasingly popular focus of NLP research. Video multimodal fusion aims to integrate the information from two or more modalities (e.g., visual and audio signals) into text for a more comprehensive reasoning. For example, multimodal sentiment analysis (Poria et al., [2020](#bib.bib22)) utilizes contrast between transcript and expression to detect sarcam, multimodal summarization (Sanabria et al., [2018](#bib.bib23)) complete summary with information only exists in visual signal.  

However, as shown in the Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"), there exist plenty of redundancy and noise in video multimodal fusion: 1) high similarity across consecutive frames brings *video redundancy*; 2) useless information, such as the distracting background, introduces *frame noise*; 3) weak alignment between visual stream and text also introduces *misalignment noise*. To alleviate the problem of redundancy and noise in video multimodal fusion, Liu et al. ([2020](#bib.bib13)) control the flow of redundant and noisy information between multimodal sequences by a fusion forget gate. The fusion forget gate impairs the impact of noise and redundancy in a coarse grain of the whole modality, so it will also filter out some representative information in the filtered modality.  

In order to remove noise and redundancy while preserving critical information in video multimodal fusion, we propose a denoising fusion bottleneck (DBF) model with mutual information maximization (MI-Max). Firstly, inspired by Nagrani et al. ([2021](#bib.bib17)), we introduce a bottleneck module to restrict the redundant and noisy information across different modalities. With the bottleneck module, inputs can only attend to low-capacity bottleneck embeddings to exchange information across different modalities, which urges redundant and noisy information to be discarded. Secondly, in order to prevent key information from being filtered out, we adopt the idea of contrastive learning to supervise the learning of our bottleneck module. Specifically, under the noise-contrastive estimation framework (Gutmann and Hyvärinen, [2010](#bib.bib5)), for each sample, we treat all the other samples in the same batch as negative ones. Then, we aim to maximize the mutual information between fusion results and each unimodal inputs by distinguishing their similarity scores from negative samples. Two aforementioned modules complement each other, the MI-Max module supervises the fusion bottleneck not to filter out key information, and in turn, the bottleneck reduces irrelevant information in fusion results to facilitate the maximization of mutual information.  

We conduct extensive experiments on three benchmarks spanning two tasks. MOSI (Zadeh et al., [2016](#bib.bib36)) and MOSEI (Zadeh et al., [2018b](#bib.bib37)) are two datasets for multimodal sentiment analysis. How2 (Sanabria et al., [2018](#bib.bib23)) is a benchmark for multimodal summarization. Experimental results show that our model achieves consistent improvements compared with current state-of-the-art methods. Meanwhile, we perform comprehensive ablation experiments to demonstrate the effectiveness of each module. In addition, we visualize the attention regions and tensity to multiple frames to intuitively show the behavior of our model to reduce noise while retaining key information implicitly.  

Concretely, we make the following contributions: (i) We propose a denoising bottleneck fusion model for video multimodal fusion, which reduces redundancy and noise while retaining key information. (ii) We achieve new state-of-the-art performance on three benchmarks spanning two video multimodal fusion tasks. (iii) We provide comprehensive ablation studies and qualitative visualization examples to demonstrate the effectiveness of both bottleneck and MI-Max modules.   

## 2 Related Work

We briefly overview related work about multimodal fusion and specific multimodal fusion tasks including multimodal summarization and multimodal sentiment analysis.  

### 2.1 Video Multimodal Fusion

Video multimodal fusion aims to join and comprehend information from two or more modalities in videos to make a comprehensive prediction. Early fusion model adopted simple network architectures. Zadeh et al. ([2017](#bib.bib34)); Liu et al. ([2018a](#bib.bib14)) fuse features by matrix operations; and Zadeh et al. ([2018a](#bib.bib35)) designed a LSTM-based model to capture both temporal and inter-modal interactions for better fusion. More recently, models influenced by prevalence of Transformer (Vaswani et al., [2017](#bib.bib29)) have emerged constantly: Zhang et al. ([2019](#bib.bib38)) injected visual information in the decoder of Transformer by cross attention mechanism to do multimodal translation task; Wu et al. ([2021](#bib.bib31)) proposed a text-centric multimodal fusion shared private framework for multimodal fusion, which consists of the cross-modal prediction and sentiment regression parts. And now vision-and-language pre-training has become a promising practice to tackle video multimodal fusion tasks. (Sun et al., [2019](#bib.bib25)) firstly extend the Transformer structure to video-language pretraining and used three pre-training tasks: masked language prediction, video text matching, masked video prediction.  

In contrast to existing works, we focus on the fundamental characteristic of video: audio and visual inputs in video are redundant and noisy (Nagrani et al., [2021](#bib.bib17)) so we aim to remove noise and redundancy while preserving critical information.  

### 2.2 Video Multimodal Summarization

Video multimodal summarization aims to generate summaries from visual features and corresponding transcripts in videos. In contrast to unimodal summarization, some information (e.g., guitar) only exists in the visual modality. Thus, for videos, utilization of both visual and text features is necessary to generate a more comprehensive summary.  

For datasets, Li et al. ([2017](#bib.bib11)) introduced a multimodal summarization dataset consisting of 500 videos of news articles in Chinese and English. Sanabria et al. ([2018](#bib.bib23)) proposed the How2 dataset consists of 2,000 hours of short instructional videos, each coming with a summary of two to three sentences.  

For models, Liu et al. ([2020](#bib.bib13)) proposed a multistage fusion network with a fusion forget gate module, which controls the flow of redundant information between multimodal long sequences. Meanwhile, Yu et al. ([2021a](#bib.bib32)) firstly introduced pre-trained language models into multimodal summarization task and experimented with the optimal injection layer of visual features.  

We also reduce redundancy in video like in (Yu et al., [2021a](#bib.bib32)). However, we do not impair the impact of noise and redundancy in a coarse grain with forget gate. Instead, we combine fusion bottleneck and MI-Max modules to filter out noise while preserving key information.  

### 2.3 Multimodal Sentiment Analysis

Multimodal sentiment analysis (MSA) aims to integrate multimodal resources, such as textual, visual, and acoustic information in videos to predict varied human emotions. In contrast to unimodal sentiment analysis, utterance in the real situation sometimes contains sarcasm, which makes it hard to make accurate prediction by a single modality. In addition, information such as expression in vision and tone in acoustic help assist sentiment prediction. Yu et al. ([2021b](#bib.bib33)) introduced a multi-label training scheme that generates extra unimodal labels for each modality and concurrently trained with the main task. Han et al. ([2021](#bib.bib6)) build up a hierarchical mutual information maximization guided model to improve the fusion outcome as well as the performance in the downstream multimodal sentiment analysis task. Luo et al. ([2021](#bib.bib16)) propose a multi-scale fusion method to align different granularity information from multiple modalities in multimodal sentiment analysis.  

Our work is fundamentally different from the above work. We do not focus on complex fusion mechanisms, but take the perspective of information in videos, and stress the importance of validity of information within fusion results.  

[FIGURE S2.F2.g1]
![Figure S2.F2.g1](./media/x2.png)

Figure 2: Overview of our denoising fusion bottleneck (DBF) model. It consists of $N$ Transformer layers to encode videos and texts, and $M$ Transformer layers with fusion bottlenecks for multimodal fusion. We incorporate a mutual information maximization (MI-Max) InfoNCE loss to regulate the bottleneck module, aiming to preserve key information in both modalities from being filtered.
[/FIGURE]

## 3 Methodology

Our denoising fusion bottleneck (DBF) model aims to fuse multimodal inputs from videos to make a comprehensive prediction. The overall architecture of DBF is shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.3 Multimodal Sentiment Analysis ‣ 2 Related Work ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"). It first employs a fusion bottleneck module with a restrained receptive field to filter out noise and redundancy when fusing different modalities in videos. Then, DBF maximizes mutual information between fusion results and unimodal inputs to supervise the learning of the fusion bottleneck, aiming to preserve more representative information in fusion results.  

### 3.1 Problem Definition

In video multimodal fusion tasks, for each video, the input comprises three sequences of encoded features from textual ($t$), visual ($v$), and acoustic ($a$) modalities. These input features are represented as $X_{m}\in\mathbb{R}^{l_{m}\times d_{m}}$, where $m\in\{t,v,a\}$, and $l_{m}$ and $d_{m}$ denote the sequence length and feature dimension for modality $m$, respectively. The goal of DBF is to extract and integrate task-related information from these input representations to form a unified fusion result $Z\in\mathbb{R}^{l\times d}$. In this paper, we evaluate the quality of the fusion result $Z$ on two tasks: video multimodal sentiment analysis and video multimodal summarization.  

For sentiment analysis, we utilize $Z$ to predict the emotional orientation of a video as a discrete category $\hat{y}$ from a predefined set of candidates $\mathcal{C}$  

|  | $$\hat{y}=\operatorname{argmax}_{y_{j}\in\mathcal{C}}\operatorname{P}_{\Theta}(y_{j}\mid Z),$$ |  | (1) |
| --- | --- | --- | --- |

or as a continuous intensity score $\hat{y}\in\mathbb{R}$  

|  | $$\hat{y}=\operatorname{P}_{\Theta}(Z),$$ |  | (2) |
| --- | --- | --- | --- |

where $\Theta$ denotes the model parameters.  

For summarization, we generate a summary sequence $\hat{S}=(s_{1},...,s_{l})$ based on $Z$:  

|  | $$\hat{S}=\text{argmax}_{S}\operatorname{P}_{\Theta}(S\mid Z).$$ |  | (3) |
| --- | --- | --- | --- |

### 3.2 Fusion Bottleneck

As shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.3 Multimodal Sentiment Analysis ‣ 2 Related Work ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"), we first employ a fusion bottleneck with a restrained receptive field to perform multimodal fusion and filter out noise and redundancy in videos. Specifically, fusion bottleneck forces cross-modal information flow passes via randomly initialized bottleneck embeddings $B\in\mathbb{R}^{{l_{b}\times d_{m}}}$ with a small sequence length, where $d_{m}$ denotes dimension of features and $l_{b}\ll l$. The restrained receptive field of $B$ forces model to collate and condense unimodal information before sharing it with the other modalities.  

With a small length $l_{b}$, embedding $B$ acts like a bottleneck in cross-modal interaction. In the fusion bottleneck module, unimodal features cannot directly attend to each other and they can only attend to the bottleneck embeddings $B$ to exchange information in it. Meanwhile, the bottleneck can attend to all of the modalities, which makes information flow across modalities must pass through the bottleneck with a restrained receptive field. The fusion bottleneck module forces the model to condense and collate information and filter out noise and redundancy.  

Specifically, in the fusion bottleneck module, with bottleneck embeddings $B$ and unimodal features $X_{m}$, the fusion result is calculated as follows:  

|  | $$[X_{m}^{l+1}||B_{m}^{l+1}]=\text{Transformer}([X_{m}^{l}||B^{l}]),$$ |  | (4) |
| --- | --- | --- | --- |

|  | $$B^{l+1}=\text{Mean}(B_{m}^{l+1}),$$ |  | (5) |
| --- | --- | --- | --- |

where $l$ denotes the layer number and $||$ denotes the concatenation operation. As shown in Equation [4](#S3.E4 "In 3.2 Fusion Bottleneck ‣ 3 Methodology ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion") and [5](#S3.E5 "In 3.2 Fusion Bottleneck ‣ 3 Methodology ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"), each time a Transformer layer is passed, bottleneck embedding $B$ is updated by unimodal features. In turn, unimodal features integrate condensed information from other modalities through bottleneck embeddings $B$. Finally, we output the text features $X_{t}^{L}$ of the last layer $L$, which are injected with condensed visual and audio information, as the fusion result.  

### 3.3 Fusion Mutual Information Maximization

The fusion bottleneck module constrains information flow across modalities in order to filter out noise and redundancy. However, it may result in loss of critical information as well when fusion bottleneck selects what information to be shared. To alleviate this issue, we employ a mutual information maximization (MI-Max) module to preserve representative and salient information from redundant modalities in fusion results.  

Mutual information is a concept from information theory that estimates the relationship between pairs of variables. Through prompting the mutual information between fusion results $Z$ and multimodal inputs $X_{m}$, we can capture modality-invariant cues among modalities (Han et al., [2021](#bib.bib6)) and keep key information preserved by regulating the fusion bottleneck module.  

Since direct maximization of mutual information for continuous and high-dimensional variables is intractable (Belghazi et al., [2018](#bib.bib1)), we instead minimize the lower bound of mutual information as Han et al. ([2021](#bib.bib6)) and Oord et al. ([2018](#bib.bib18)). To be specific, we first construct an opposite path from $Z$ to predict $X_{m}$ by an MLP $\mathcal{F}$. Then, to gauge correlation between the prediction and $X_{m}$, we use a normalized similarity function as follows:  

|  | $$\text{sim}(X_{m},Z)=\text{exp}\left(\frac{X_{m}}{\left\|X_{m}\right\|^{2}}\odot\frac{\mathcal{F}(Z)}{\left\|\mathcal{F}(Z)\right\|^{2}}\right),$$ |  | (6) |
| --- | --- | --- | --- |

where $\mathcal{F}$ generates a prediction of $X_{m}$ from $Z$, $\|\cdot\|^{2}$ is the Euclidean norm, and $\odot$ denotes element-wise product. Then, we incorporate this similarity function into the noise-contrastive estimation framework (Gutmann and Hyvärinen, [2010](#bib.bib5)) and produce an InfoNCE loss (Oord et al., [2018](#bib.bib18)) which reflects the lower bound of the mutual information:  

|  | $$\mathcal{L}_{\text{NCE}}^{z,m}=-\mathbb{E}_{X_{m},Z}\left[\log\frac{e^{\operatorname{sim}\left(x_{m}^{{+}},\mathcal{F}(Z)\right)}}{\sum_{k=1}^{K}e^{\operatorname{sim}\left(\tilde{x}_{m}^{k},\mathcal{F}(Z)\right)}}\right]$$ |  | (7) |
| --- | --- | --- | --- |

where $\tilde{x}_{m}=\left\{\tilde{x}^{1},\ldots,\tilde{x}^{K}\right\}$ is the negative unimodal inputs that are not matched to the fusion result $Z$ in same batch. Finally, we compute loss for all modalities as follows:  

|  | $$\mathcal{L}_{\text{NCE}}=\alpha(\mathcal{L}_{\text{NCE}}^{z,v}+\mathcal{L}_{\text{NCE}}^{z,a}+\mathcal{L}_{\text{NCE}}^{z,t})$$ |  | (8) |
| --- | --- | --- | --- |

where $\alpha$ is a hyper-parameter that controls the impact of MI-Max.  

By minimizing $\mathcal{L}_{\text{NCE}}$, on the one hand, we maximize the lower bound of the mutual information between fusion results and unimodal inputs; on the other hand, we encourage fusion results to reversely predict unimodal inputs as well as possible, which prompts retaining of representative and key information from different modalities in fusion results.  

[TABLE S3.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">MOSI</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">MAE(<math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">Corr(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">Acc-7(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">Acc-2(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">F1(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">MulT <cite class="ltx_cite ltx_citemacro_citep">(Tsai et al., <a class="ltx_ref">2019</a>)</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">0.871</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.698</td>
<td class="ltx_td ltx_align_center ltx_border_t">40.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">- / 83.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">- / 82.8</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">TFN <cite class="ltx_cite ltx_citemacro_citep">(Zadeh et al., <a class="ltx_ref">2017</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.901</td>
<td class="ltx_td ltx_align_center">0.698</td>
<td class="ltx_td ltx_align_center">34.9</td>
<td class="ltx_td ltx_align_center">- / 80.8</td>
<td class="ltx_td ltx_align_center">- / 80.7</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">LMF <cite class="ltx_cite ltx_citemacro_citep">(Liu et al., <a class="ltx_ref">2018b</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.917</td>
<td class="ltx_td ltx_align_center">0.695</td>
<td class="ltx_td ltx_align_center">33.2</td>
<td class="ltx_td ltx_align_center">- / 82.5</td>
<td class="ltx_td ltx_align_center">- / 82.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MFM <cite class="ltx_cite ltx_citemacro_citep">(Tsai et al., <a class="ltx_ref">2018</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.877</td>
<td class="ltx_td ltx_align_center">0.706</td>
<td class="ltx_td ltx_align_center">35.4</td>
<td class="ltx_td ltx_align_center">- / 81.7</td>
<td class="ltx_td ltx_align_center">- / 81.6</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ICCN <cite class="ltx_cite ltx_citemacro_citep">(Sun et al., <a class="ltx_ref">2020</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.860</td>
<td class="ltx_td ltx_align_center">0.710</td>
<td class="ltx_td ltx_align_center">39.0</td>
<td class="ltx_td ltx_align_center">- / 83.0</td>
<td class="ltx_td ltx_align_center">- / 83.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MISA <cite class="ltx_cite ltx_citemacro_citep">(Hazarika et al., <a class="ltx_ref">2020</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.783</td>
<td class="ltx_td ltx_align_center">0.761</td>
<td class="ltx_td ltx_align_center">42.3</td>
<td class="ltx_td ltx_align_center">81.8 / 83.4</td>
<td class="ltx_td ltx_align_center">81.7 / 83.6</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Self-MM <cite class="ltx_cite ltx_citemacro_citep">(Yu et al., <a class="ltx_ref">2021b</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.712</td>
<td class="ltx_td ltx_align_center">0.795</td>
<td class="ltx_td ltx_align_center">45.8</td>
<td class="ltx_td ltx_align_center">82.5 / 84.8</td>
<td class="ltx_td ltx_align_center">82.7 / 84.9</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MMIM<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">†</span></sup> <cite class="ltx_cite ltx_citemacro_citep">(Han et al., <a class="ltx_ref">2021</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.700</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.800</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">46.7</span></td>
<td class="ltx_td ltx_align_center">84.2 / 86.1</td>
<td class="ltx_td ltx_align_center">84.0 / 86.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">DBF</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.693</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.801</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">44.8</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">85.1 / 86.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">85.1 / 86.9</span></td>
</tr>
</tbody>
</table>

Table 1: Results of multimodal sentiment analysis on MOSI. ${\dagger}$ indicates the previous state-of-the-art model.
[/TABLE]

[TABLE S3.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">MOSEI</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">MAE(<math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">Corr(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">Acc-7(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">Acc-2(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">F1(<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">MulT <cite class="ltx_cite ltx_citemacro_citep">(Tsai et al., <a class="ltx_ref">2019</a>)</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">0.580</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.703</td>
<td class="ltx_td ltx_align_center ltx_border_t">51.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">- / 82.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">- / 82.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">TFN <cite class="ltx_cite ltx_citemacro_citep">(Zadeh et al., <a class="ltx_ref">2017</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.593</td>
<td class="ltx_td ltx_align_center">0.700</td>
<td class="ltx_td ltx_align_center">50.2</td>
<td class="ltx_td ltx_align_center">- / 82.1</td>
<td class="ltx_td ltx_align_center">- / 82.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">LMF <cite class="ltx_cite ltx_citemacro_citep">(Liu et al., <a class="ltx_ref">2018b</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.677</td>
<td class="ltx_td ltx_align_center">0.695</td>
<td class="ltx_td ltx_align_center">48.0</td>
<td class="ltx_td ltx_align_center">- / 82.1</td>
<td class="ltx_td ltx_align_center">- / 82.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MFM <cite class="ltx_cite ltx_citemacro_citep">(Tsai et al., <a class="ltx_ref">2018</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.717</td>
<td class="ltx_td ltx_align_center">0.706</td>
<td class="ltx_td ltx_align_center">51.3</td>
<td class="ltx_td ltx_align_center">- / 84.3</td>
<td class="ltx_td ltx_align_center">- / 84.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ICCN <cite class="ltx_cite ltx_citemacro_citep">(Sun et al., <a class="ltx_ref">2020</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.565</td>
<td class="ltx_td ltx_align_center">0.713</td>
<td class="ltx_td ltx_align_center">51.6</td>
<td class="ltx_td ltx_align_center">- / 84.2</td>
<td class="ltx_td ltx_align_center">- / 84.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MISA <cite class="ltx_cite ltx_citemacro_citep">(Hazarika et al., <a class="ltx_ref">2020</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.555</td>
<td class="ltx_td ltx_align_center">0.756</td>
<td class="ltx_td ltx_align_center">52.2</td>
<td class="ltx_td ltx_align_center">83.8 / 85.3</td>
<td class="ltx_td ltx_align_center">83.6 / 85.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Self-MM <cite class="ltx_cite ltx_citemacro_citep">(Yu et al., <a class="ltx_ref">2021b</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.529</td>
<td class="ltx_td ltx_align_center">0.767</td>
<td class="ltx_td ltx_align_center">53.5</td>
<td class="ltx_td ltx_align_center">82.7 / 85.0</td>
<td class="ltx_td ltx_align_center">83.0 / 84.9</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MMIM<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">†</span></sup> <cite class="ltx_cite ltx_citemacro_citep">(Han et al., <a class="ltx_ref">2021</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">0.526</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.772</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">54.2</span></td>
<td class="ltx_td ltx_align_center">82.2 / 86.0</td>
<td class="ltx_td ltx_align_center">82.7 / 85.9</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">DBF</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.523</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.772</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">54.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">84.3 / 86.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">84.8 / 86.2</span></td>
</tr>
</tbody>
</table>

Table 2: Results of multimodal sentiment analysis on MOSEI. ${\dagger}$ indicates the previous state-of-the-art model.
[/TABLE]

## 4 Experiments

### 4.1 Tasks, Datasets, and Metrics

We evaluate fusion results of DBF on two video multimodal tasks: video multimodal sentiment analysis and video multimodal summarization.  

#### Video Multimodal Sentiment Analysis

Video multimodal sentiment analysis is a regression task that aims to collect and tackle data from multiple resources (text, vision and acoustic) to comprehend varied human emotions. We do this task on MOSI (Zadeh et al., [2016](#bib.bib36)) and MOSEI (Zadeh et al., [2018b](#bib.bib37)) datasets. The MOSI dataset contains 2198 subjective utterance-video segments, which are manually annotated with a continuous opinion score between [-3, 3], where -3/+3 represents strongly negative/positive sentiments. The MOSEI dataset is an improvement over MOSI, which contains 23453 annotated video segments (utterances), from 5000 videos, 1000 distinct speakers and 250 different topics.  

Following (Hazarika et al., [2020](#bib.bib8)), we use the same metric set to evaluate sentiment intensity predictions: MAE (mean absolute error), which is the average of absolute difference value between predictions and labels; Corr (Pearson correlation) that measures the degree of prediction skew; Acc-7 (seven-class classification accuracy) ranging from -3 to 3; Acc-2 (binary classification accuracy) and F1 score computed for positive/negative and non-negative/negative classification results.  

#### Video Multimodal Summarization

The summary task aims to generate abstractive summarization with videos and their corresponding transcripts. We set How2 dataset (Sanabria et al., [2018](#bib.bib23)) as benchmark for this task, which is a large-scale dataset consists of 79,114 short instructional videos, and each video is accompanied by a human-generated transcript and a short text summary.  

Following (Yu et al., [2021a](#bib.bib32)), to evaluate summarization, we use metrics as follows: ROUGE (Lin and Hovy, [2003](#bib.bib12)) (ROUGE-1, 2, L) and BLEU (Papineni et al., [2002](#bib.bib20)) (BLEU-1, 2, 3, 4), which calculate the recall and precision of n-gram overlaps, respectively; METEOR (Denkowski and Lavie, [2011](#bib.bib3)), which evaluates matching degree of word stems, synonyms and paraphrases; CIDEr (Vedantam et al., [2015](#bib.bib30)) is an image captioning metric to compute the cosine similarity between TF-IDF weighted n-grams.  

### 4.2 Experimental Settings

For sentiment analysis task, we use BERT-base (Devlin et al., [2018](#bib.bib4)) to encode text input and extract the [CLS] embedding from the last layer. For acoustic and vision, we use COVAREP (Degottex et al., [2014](#bib.bib2)) and Facet 111https://imotions.com/platform/ to extract audio and facial expression features. The visual feature dimensions are 47 for MOSI, 35 for MOSEI, and the audio feature dimensions are 74 for both MOSI and MOSEI.  

For summarization, we use BART (Lewis et al., [2019](#bib.bib10)) as the feature extractor and inject visual information in the last layer of the BART encoder. For vision, a 2048-dimensional feature representation is extracted for every 16 non-overlapping frames using a 3D ResNeXt-101 model (Hara et al., [2018](#bib.bib7)), which is pre-trained on the Kinetics dataset (Kay et al., [2017](#bib.bib9)). Details of the hyper-parameters are given in Appendix [A](#A1 "Appendix A Hyper-parameters ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"). For frameworks and hardware, we use the deep learning framework PyTorch (Paszke et al., [2017](#bib.bib21)) and Huggingface 222https://huggingface.co/ to implement our code. We use a single Nvidia GeForce A40 GPU for sentiment analysis experiments and two for summarization.  

### 4.3 Overall Results

[TABLE S4.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">How2</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">R-1</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">R-2</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">R-L</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">B-1</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">B-2</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">B-3</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">B-4</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">METEOR</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">CIDEr</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">HA (RNN) <cite class="ltx_cite ltx_citemacro_citep">(Palaskar et al., <a class="ltx_ref">2019</a>)</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">60.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">42.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">55.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">57.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">47.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">41.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">37.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">28.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.48</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">HA (TF) <cite class="ltx_cite ltx_citemacro_citep">(Palaskar et al., <a class="ltx_ref">2019</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">60.2</td>
<td class="ltx_td ltx_align_center">43.1</td>
<td class="ltx_td ltx_align_center">55.9</td>
<td class="ltx_td ltx_align_center">58.6</td>
<td class="ltx_td ltx_align_center">48.3</td>
<td class="ltx_td ltx_align_center">43.3</td>
<td class="ltx_td ltx_align_center">38.1</td>
<td class="ltx_td ltx_align_center">28.9</td>
<td class="ltx_td ltx_align_center">2.51</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MFFG (RNN) <cite class="ltx_cite ltx_citemacro_citep">(Liu et al., <a class="ltx_ref">2020</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">62.3</td>
<td class="ltx_td ltx_align_center">46.1</td>
<td class="ltx_td ltx_align_center">58.2</td>
<td class="ltx_td ltx_align_center">59.1</td>
<td class="ltx_td ltx_align_center">50.4</td>
<td class="ltx_td ltx_align_center">45.1</td>
<td class="ltx_td ltx_align_center">41.1</td>
<td class="ltx_td ltx_align_center">30.1</td>
<td class="ltx_td ltx_align_center">2.69</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">MFFG (TF) <cite class="ltx_cite ltx_citemacro_citep">(Liu et al., <a class="ltx_ref">2020</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">61.6</td>
<td class="ltx_td ltx_align_center">45.1</td>
<td class="ltx_td ltx_align_center">57.4</td>
<td class="ltx_td ltx_align_center">60.0</td>
<td class="ltx_td ltx_align_center">50.9</td>
<td class="ltx_td ltx_align_center">45.3</td>
<td class="ltx_td ltx_align_center">41.3</td>
<td class="ltx_td ltx_align_center">29.9</td>
<td class="ltx_td ltx_align_center">2.67</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">VG-GPLMs<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">†</span></sup> <cite class="ltx_cite ltx_citemacro_citep">(Yu et al., <a class="ltx_ref">2021a</a>)</cite>
</th>
<td class="ltx_td ltx_align_center">68.0</td>
<td class="ltx_td ltx_align_center">51.4</td>
<td class="ltx_td ltx_align_center">63.3</td>
<td class="ltx_td ltx_align_center">65.2</td>
<td class="ltx_td ltx_align_center">56.3</td>
<td class="ltx_td ltx_align_center">50.4</td>
<td class="ltx_td ltx_align_center">46.0</td>
<td class="ltx_td ltx_align_center">34.0</td>
<td class="ltx_td ltx_align_center">3.28</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">DBF</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">70.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">54.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">66.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">67.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">58.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">53.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">49.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">35.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">3.56</span></td>
</tr>
</tbody>
</table>

Table 3: Results of multimodal summarization task on How2. The ${\dagger}$ indicates the previous state-of-the-art model. We denote ROUGE and BLEU by R and B respectively.
[/TABLE]

[TABLE S4.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">MOSI</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt">
<span class="ltx_text ltx_font_bold">MOSEI</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">MAE (<math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>)</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">F1 (<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">MAE (<math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>)</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">F1 (<math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>)</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">1) Ours</th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.693</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">85.07 / 86.88</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.523</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">84.78 / 86.19</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">2) (-) MI-Max</th>
<td class="ltx_td ltx_align_center">0.697</td>
<td class="ltx_td ltx_align_center ltx_border_r">83.08 / 85.28</td>
<td class="ltx_td ltx_align_center">0.536</td>
<td class="ltx_td ltx_align_center">80.94 / 85.58</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">3) (-) bottleneck</th>
<td class="ltx_td ltx_align_center">0.750</td>
<td class="ltx_td ltx_align_center ltx_border_r">82.84 / 83.63</td>
<td class="ltx_td ltx_align_center">0.537</td>
<td class="ltx_td ltx_align_center">77.52 / 83.81</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">4) (-) Language <math class="ltx_Math"><semantics><mi>l</mi><annotation-xml><ci>𝑙</ci></annotation-xml><annotation>l</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">1.391</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">55.54 / 54.95</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.817</td>
<td class="ltx_td ltx_align_center ltx_border_t">67.63 / 64.01</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">5) (-) Visual <math class="ltx_Math"><semantics><mi>v</mi><annotation-xml><ci>𝑣</ci></annotation-xml><annotation>v</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center">0.700</td>
<td class="ltx_td ltx_align_center ltx_border_r">82.78 / 84.33</td>
<td class="ltx_td ltx_align_center">0.541</td>
<td class="ltx_td ltx_align_center">78.42 / 84.05</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">6) (-) Audio <math class="ltx_Math"><semantics><mi>a</mi><annotation-xml><ci>𝑎</ci></annotation-xml><annotation>a</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center">0.720</td>
<td class="ltx_td ltx_align_center ltx_border_r">83.02 / 85.86</td>
<td class="ltx_td ltx_align_center">0.536</td>
<td class="ltx_td ltx_align_center">80.22 / 85.02</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">7) Visual-based</th>
<td class="ltx_td ltx_align_center ltx_border_t">1.372</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">57.06 / 57.83</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.536</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.41 / 85.47</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">8) Audio-based</th>
<td class="ltx_td ltx_align_center ltx_border_bb">1.194</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">67.95 / 70.49</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.537</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.80 / 85.76</td>
</tr>
</tbody>
</table>

Table 4: Results of ablation study. (-) represents removal for the mentioned factors.
Model 1 represents the best performing model in each dataset; Model 2,3 presents the effect of MI module and bottleneck module; Model 4,5,6 depicts the effect of individual modalities; Model 7,8 presents the variants of our model as defined in Section [4.4](#S4.SS4 "4.4 Ablation Study ‣ 4 Experiments ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion").
[/TABLE]

[FIGURE S4.F3.g1]
![Figure S4.F3.g1](./media/x3.png)

Figure 3: Comparison of Grad-CAM visualizations of baseline method VG-GPLMs (Yu et al., [2021a](#bib.bib32)) (top) and DBF (bottom).
In contrast to even attention to different frames of the baseline method, DBF ignores redundancy and noise in consecutive frames and highly focuses on the key information (*pouring wine* in this example) in a particular frame.
[/FIGURE]

We compare performance against DBF by considering various baselines as below: For multimodal sentiment analysis, we compare with MulT (Tsai et al., [2019](#bib.bib27)), TFN (Zadeh et al., [2017](#bib.bib34)), LMF (Liu et al., [2018b](#bib.bib15)), MFM (Tsai et al., [2018](#bib.bib28)), ICCN (Sun et al., [2020](#bib.bib26)), MISA (Hazarika et al., [2020](#bib.bib8)), Self-MM (Yu et al., [2021b](#bib.bib33)) and MMIM (Han et al., [2021](#bib.bib6)). For multimodal summarization, we compare with HA (Palaskar et al., [2019](#bib.bib19)) MFFG (Liu et al., [2020](#bib.bib13)) VG-GPLMs (Yu et al., [2021a](#bib.bib32)). Details of baselines are in Appendix [B](#A2 "Appendix B Baselines ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"). The comparative results for sentiment analysis are presented in Table [1](#S3.T1 "Table 1 ‣ 3.3 Fusion Mutual Information Maximization ‣ 3 Methodology ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion") (MOSI) and Table [2](#S3.T2 "Table 2 ‣ 3.3 Fusion Mutual Information Maximization ‣ 3 Methodology ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion") (MOSEI). Results for summarization are presented in Table [3](#S4.T3 "Table 3 ‣ 4.3 Overall Results ‣ 4 Experiments ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion") (How2).  

We find that DBF yields better or comparable results to state-of-the-art methods. To elaborate, DBF significantly outperforms state-of-the-art in all metrics on How2 and in most of metrics on MOSI and MOSEI. For other metrics, DBF achieves very closed performance to state-of-the-art. These outcomes preliminarily demonstrate the efficacy of our method in video multimodal fusion.  

From the results, we can observe that our model achieves more significant performance improvement on summary task than sentiment analysis. There could be two reasons for this: 1) the size of two datasets is small, yet DBF requires a sufficient amount of data to learn noise and redundancy patterns for this type of video. 2) Visual features are extracted by Facet on sentiment analysis task and more 3D ResNeXt-101 on summary task respectively. Compared to sentiment analysis task, summary task employ a more advanced visual extractor and DBF is heavily influenced by the quality of visual features.  

### 4.4 Ablation Study

#### Effect of Fusion Bottleneck and MI-Max

As shown in Table [4](#S4.T4 "Table 4 ‣ 4.3 Overall Results ‣ 4 Experiments ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"), we first remove respectively MI-Max module and exchange fusion bottleneck module with vanilla fusion methods to observe the effects on performance. We observe that fusion bottleneck and MI-Max both help better fusion results, and the combination of them further improves performance, which reflects the necessity of removing noise while maintaining representative information.  

#### Effect of Modalities

Then we remove one modality at a time to observe the effect on performance. Firstly, we observe that the multimodal combination provides the best performance, indicating that our model can learn complementary information from different modalities. Next, we observe that the performance drops sharply when the language modality is removed. This may be due to the fact that text has higher information density compared to redundant audio and visual modalities. It verifies two things: 1) It is critical to remove noise and redundancy to increase information density of visual and audio modalities when doing fusion. 2) Text-centric fusion results may help improve performance on multimodal summary and sentiment analysis tasks.  

#### Effect of Center Modality

As mentioned above, text-centric fusion results tend to perform better as low information intensity and high redundancy in other modalities. Thus, we evaluate fusion results based on acoustic and vision modality respectively on downstream tasks. We observe an obvious decline in performance when audio or visual modality is used as the central modality.  

### 4.5 Case Study

In this section, we first calculate standard deviation and normalized entropy over visual attention scores in the Grad-CAM heatmaps (Selvaraju et al., [2017](#bib.bib24)) for DBF and baseline method VG-GPLMs (Yu et al., [2021a](#bib.bib32)) respectively. These two metrics show the sharpness of visual attention scores, indicating whether the model focuses more on key frames and ignores redundant content. Then, we compute visualizations on Grad-CAM heatmaps acquired before to show the ability of DBF to filter out redundancy and preserve key information.  

#### Statistics of Visualization Results

Grad-CAM is a visualization method of images, it obtains visualization heatmaps by calculating weights and gradients during backpropagation, and in this paper we extend Grad-CAM to videos. Further, to quantify this sharpness of visual attention, we calculate standard deviation and normalized entropy on Grad-CAM heatmaps over the test split on How2 dataset. For results, DBF gets 0.830, 0.008, baseline gets 0.404, 0.062 in deviation and normalized entropy respectively. DBF holds a higher deviation and lower entropy, which indicates sharper visual attention maps to discriminate redundancy and key frames.   

#### Visualization Example

Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Overall Results ‣ 4 Experiments ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion") provides Grad-CAM visualizations of DBF and baseline method. As we can see, DBF has more sharp attention over continuous frames and ignores redundancy while preserving critical information in visual inputs.  

## 5 Conclusion

In this paper, we propose a denoising video multimodal fusion system DBF which contains a fusion bottleneck to filter out redundancy with noise, a mutual information module to preserve key information in fusion results. Our model alleviates redundancy and nosie problem in video multimodal fusion and makes full use of all representative information in redundant modalities (vision and acoustic). In the experiments, we show that our model significantly and consistently outperforms state-of-the-art video multimodal models. In addition, we demonstrate that DBF can appropriately select necessary contents and neglect redundancy in video by comprehensive ablation and visualization studies.  

In the future, we will explore the following directions: (1) We will try to extend the proposed DBF model to more multimodal fusion tasks such as humor detection. (2) We will incorporate vision-text pretraining backbones into our DBF model to further improve its performance.  

## Limitations

First, limited by the category of video multimodal fusion tasks, we do not perform experiments on more tasks to better validate the effectiveness of our method, and we hope to extend our model to more various and complete benchmarks in future work. Secondly, as shown in Section [4.3](#S4.SS3 "4.3 Overall Results ‣ 4 Experiments ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion"), our model achieves relatively slight performance improvement on sentiment analysis task. For reasons, our model may be dependent on the scale of datasets to learn noise and redundancy patterns in video, which needs to be further improved and studied.  

## Acknowledgement

This paper is supported by the National Key Research and Development Program of China 2020AAA0106700 and NSFC project U19A2065.  

## References

* Belghazi et al. (2018)  Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and R Devon Hjelm. 2018.   Mine: mutual information neural estimation.   *arXiv preprint arXiv:1801.04062*. 
* Degottex et al. (2014)  Gilles Degottex, John Kane, Thomas Drugman, Tuomo Raitio, and Stefan Scherer. 2014.   Covarep—a collaborative voice analysis repository for speech technologies.   In *2014 ieee international conference on acoustics, speech and signal processing (icassp)*, pages 960–964. IEEE. 
* Denkowski and Lavie (2011)  Michael Denkowski and Alon Lavie. 2011.   Meteor 1.3: Automatic metric for reliable optimization and evaluation of machine translation systems.   In *Proceedings of the sixth workshop on statistical machine translation*, pages 85–91. 
* Devlin et al. (2018)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018.   Bert: Pre-training of deep bidirectional transformers for language understanding.   *arXiv preprint arXiv:1810.04805*. 
* Gutmann and Hyvärinen (2010)  Michael Gutmann and Aapo Hyvärinen. 2010.   Noise-contrastive estimation: A new estimation principle for unnormalized statistical models.   In *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, pages 297–304. JMLR Workshop and Conference Proceedings. 
* Han et al. (2021)  Wei Han, Hui Chen, and Soujanya Poria. 2021.   Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis.   *arXiv preprint arXiv:2109.00412*. 
* Hara et al. (2018)  Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. 2018.   Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet?   In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, pages 6546–6555. 
* Hazarika et al. (2020)  Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria. 2020.   Misa: Modality-invariant and-specific representations for multimodal sentiment analysis.   In *Proceedings of the 28th ACM international conference on multimedia*, pages 1122–1131. 
* Kay et al. (2017)  Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. 2017.   The kinetics human action video dataset.   *arXiv preprint arXiv:1705.06950*. 
* Lewis et al. (2019)  Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2019.   Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.   *arXiv preprint arXiv:1910.13461*. 
* Li et al. (2017)  Haoran Li, Junnan Zhu, Cong Ma, Jiajun Zhang, and Chengqing Zong. 2017.   Multi-modal summarization for asynchronous collection of text, image, audio and video.   In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 1092–1102. 
* Lin and Hovy (2003)  Chin-Yew Lin and Eduard Hovy. 2003.   Automatic evaluation of summaries using n-gram co-occurrence statistics.   In *Proceedings of the 2003 human language technology conference of the North American chapter of the association for computational linguistics*, pages 150–157. 
* Liu et al. (2020)  Nayu Liu, Xian Sun, Hongfeng Yu, Wenkai Zhang, and Guangluan Xu. 2020.   Multistage fusion with forget gate for multimodal summarization in open-domain videos.   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1834–1845. 
* Liu et al. (2018a)  Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan, Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. 2018a.   Efficient low-rank multimodal fusion with modality-specific factors.   *arXiv preprint arXiv:1806.00064*. 
* Liu et al. (2018b)  Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan, Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. 2018b.   Efficient low-rank multimodal fusion with modality-specific factors.   *arXiv preprint arXiv:1806.00064*. 
* Luo et al. (2021)  Huaishao Luo, Lei Ji, Yanyong Huang, Bin Wang, Shenggong Ji, and Tianrui Li. 2021.   Scalevlad: Improving multimodal sentiment analysis via multi-scale fusion of locally descriptors.   *arXiv preprint arXiv:2112.01368*. 
* Nagrani et al. (2021)  Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, and Chen Sun. 2021.   Attention bottlenecks for multimodal fusion.   *Advances in Neural Information Processing Systems*, 34:14200–14213. 
* Oord et al. (2018)  Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.   Representation learning with contrastive predictive coding.   *arXiv preprint arXiv:1807.03748*. 
* Palaskar et al. (2019)  Shruti Palaskar, Jindrich Libovickỳ, Spandana Gella, and Florian Metze. 2019.   Multimodal abstractive summarization for how2 videos.   *arXiv preprint arXiv:1906.07901*. 
* Papineni et al. (2002)  Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.   Bleu: a method for automatic evaluation of machine translation.   In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pages 311–318. 
* Paszke et al. (2017)  A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. Devito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. 2017.   Automatic differentiation in pytorch. 
* Poria et al. (2020)  Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, and Rada Mihalcea. 2020.   Beneath the tip of the iceberg: Current challenges and new directions in sentiment analysis research.   *IEEE Transactions on Affective Computing*. 
* Sanabria et al. (2018)  Ramon Sanabria, Ozan Caglayan, Shruti Palaskar, Desmond Elliott, Loïc Barrault, Lucia Specia, and Florian Metze. 2018.   How2: a large-scale dataset for multimodal language understanding.   *arXiv preprint arXiv:1811.00347*. 
* Selvaraju et al. (2017)  Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. 2017.   Grad-cam: Visual explanations from deep networks via gradient-based localization.   In *Proceedings of the IEEE international conference on computer vision*, pages 618–626. 
* Sun et al. (2019)  Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. 2019.   Videobert: A joint model for video and language representation learning.   In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 
* Sun et al. (2020)  Zhongkai Sun, Prathusha Sarma, William Sethares, and Yingyu Liang. 2020.   Learning relationships between text, audio, and video via deep canonical correlation for multimodal language analysis.   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 8992–8999. 
* Tsai et al. (2019)  Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019.   Multimodal transformer for unaligned multimodal language sequences.   In *Proceedings of the conference. Association for Computational Linguistics. Meeting*, volume 2019, page 6558. NIH Public Access. 
* Tsai et al. (2018)  Yao-Hung Hubert Tsai, Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2018.   Learning factorized multimodal representations.   *arXiv preprint arXiv:1806.06176*. 
* Vaswani et al. (2017)  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.   Attention is all you need.   *Advances in neural information processing systems*, 30. 
* Vedantam et al. (2015)  Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015.   Cider: Consensus-based image description evaluation.   In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4566–4575. 
* Wu et al. (2021)  Yang Wu, Zijie Lin, Yanyan Zhao, Bing Qin, and Li-Nan Zhu. 2021.   A text-centered shared-private framework via cross-modal prediction for multimodal sentiment analysis.   In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 4730–4738. 
* Yu et al. (2021a)  Tiezheng Yu, Wenliang Dai, Zihan Liu, and Pascale Fung. 2021a.   Vision guided generative pre-trained language models for multimodal abstractive summarization.   *arXiv preprint arXiv:2109.02401*. 
* Yu et al. (2021b)  Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu. 2021b.   Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis.   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 10790–10797. 
* Zadeh et al. (2017)  Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2017.   Tensor fusion network for multimodal sentiment analysis.   *arXiv preprint arXiv:1707.07250*. 
* Zadeh et al. (2018a)  Amir Zadeh, Paul Pu Liang, Navonil Mazumder, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018a.   Memory fusion network for multi-view sequential learning.   In *Proceedings of the AAAI conference on artificial intelligence*, volume 32. 
* Zadeh et al. (2016)  Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. 2016.   Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos.   *arXiv preprint arXiv:1606.06259*. 
* Zadeh et al. (2018b)  AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018b.   Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph.   In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2236–2246. 
* Zhang et al. (2019)  Zhuosheng Zhang, Kehai Chen, Rui Wang, Masao Utiyama, Eiichiro Sumita, Zuchao Li, and Hai Zhao. 2019.   Neural machine translation with universal visual representation.   In *International Conference on Learning Representations*. 

## Appendix

## Appendix A Hyper-parameters

We set hyper-parameters as shown in Table [5](#A1.T5 "Table 5 ‣ Appendix A Hyper-parameters ‣ Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion") for best performance. For optimization, we utilize the Adam optimizer with warmup. The training duration of each model is governed by early-stopping strategy with a patience of 10 epochs.  

[TABLE A1.T5]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Hyper-Parameter</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">MOSI</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">MOSEI</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">How2</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_t">Batch size</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">32</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">96</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">80</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">Bottleneck length</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">2</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">4</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">Num of bottleneck layers</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">4</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">4</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left"><math class="ltx_Math"><semantics><mi>α</mi><annotation-xml><ci>𝛼</ci></annotation-xml><annotation>\alpha</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">0.05</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">0.1</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">0.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">Learning rate <math class="ltx_Math"><semantics><msub><mi>η</mi><mtext>DBF</mtext></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝜂</ci><ci><mtext>DBF</mtext></ci></apply></annotation-xml><annotation>\eta_{\text{DBF}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">2e-05</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">2e-03</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">3e-04</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">Learning rate <math class="ltx_Math"><semantics><msub><mi>η</mi><mtext>Backbone</mtext></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝜂</ci><ci><mtext>Backbone</mtext></ci></apply></annotation-xml><annotation>\eta_{\text{Backbone}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">1e-04</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">5e-05</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">6e-05</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_bb">Fusion size</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">128</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">128</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">768</td>
</tr>
</tbody>
</table>

Table 5: Hyper-parameters for the best performance.
$\eta_{\text{Backbone}}$ denotes the learning rate of parameters of the backbone pretrained model.
$\eta_{\text{DBF}}$ denotes the learning rate of new parameters introduced by our DBF model.
[/TABLE]

## Appendix B Baselines

For multimodal sentiment analysis:  

#### MulT (Tsai et al., [2019](#bib.bib27)) :

a multimodal transformer architecture model with directional pairwise cross-attention, which translates one modality to another.  

#### TFN (Zadeh et al., [2017](#bib.bib34))

based on tensor outer product to capture multiple-modal interactions.  

#### LMF (Liu et al., [2018b](#bib.bib15)) :

an advanced version of TFN model.  

#### MFM (Tsai et al., [2018](#bib.bib28)) :

a model that factorizes representations into two sets of independent factors: multimodal discriminative and modality-specific generative factors.  

#### ICCN (Sun et al., [2020](#bib.bib26)) :

an adversarial encoder-decoder classifier framework-based model to learn a modality-invariant embedding space.  

#### MISA (Hazarika et al., [2020](#bib.bib8))

projects each modality to two distinct subspaces.  

#### Self-MM (Yu et al., [2021b](#bib.bib33))

propose a label generation module based on the self-supervised learning strategy to acquire independent unimodal supervision.  

#### MMIM (Han et al., [2021](#bib.bib6))

hierarchically maximizes the mutual information in unimodal input pairs and between multimodal fusion result and unimodal input.  

For multimodal summarization, We compare DBF with the following baselines:  

#### HA (Palaskar et al., [2019](#bib.bib19)) :

a sequence-to-sequence multimodal fusion model with hierarchical attention.  

#### MFFG (Liu et al., [2020](#bib.bib13)) :

a multistage fusion network with the fusion forget gate module, which controls the flow of redundant information between multimodal long sequences via a forgetting module.  

#### VG-GPLMs (Yu et al., [2021a](#bib.bib32)) :

a BART-based and vision guided model for multimodal summarization task, which use attention-based add-on layers to incorporate visual information.  

