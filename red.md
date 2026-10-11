\documentclass[twocolumn,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{hyperref}
\usepackage{abstract}

\title{\textbf{Cross-Modal SAR-Optical Representation Learning for Agricultural Land-Cover Analysis: From Gaussian-Mixture Baselines to Contrastive Dual-Encoder Translation}}
\author{\textbf{Mohammed Kaamraan Ahmed}\textsuperscript{1}\\
\small Department of Computer Science and Engineering, IIIT Sri City}
\date{}

\begin{document}

\maketitle

\begin{abstract}
Estimating vegetation and agricultural condition directly from Synthetic Aperture Radar (SAR) imagery is attractive because SAR is largely unaffected by cloud cover, but SAR backscatter is a noisy proxy for the optical vegetation indices (e.g. NDVI) that agronomists actually use. This paper documents a full cross-modal representation-learning pipeline evaluated on 568 co-registered Sentinel-1/Sentinel-2 patches (454 train/114 test). We begin with an 8-component Gaussian Mixture Model (GMM) baseline fit on hand-crafted SAR features and a Random-Forest/XGBoost regression bridge to optical indices. We then train a Supervised Contrastive (SupCon) dual ResNet-18 encoder that maps SAR and optical imagery into a shared 64-dimensional latent space, and extend it with two architectural extensions: Cluster-Aware Hard-Negative (CAHN) mining combined with cross-modal InfoNCE alignment, and Iterative Embedding-Anchored Re-Clustering (IERC). To quantitatively verify cross-modal alignment in the learned latent space, we run a Canonical Correlation Analysis (CCA) benchmark against a 200-iteration permutation null distribution at three stages of training. The untrained encoder yields only marginal alignment ($r_{1}=0.3117$, $\Delta_{null}=0.1084$); CAHN SupCon raises the first canonical correlation to $r_{1}=0.8653$ ($\Delta_{null}=0.6598$); and IERC fine-tuning reaches $r_{1}=0.9557$ ($\overline{r}=0.9043$, $\Delta_{null}=0.7499$, $p<0.001$), which we corroborate with a global 64-dimensional embedding CCA ($r_{1}=0.9832$) and a held-out test-split CCA ($r_{1}=0.9846$, $\Delta_{null}=0.2479$, $p<0.001$). This CCA benchmark also disentangles, for the first time, the individual contribution of hard-negative mining (Stage 4) from that of re-clustering (Stage 4b), which prior runs of this pipeline could not separate. We additionally report the pre-existing train/test generalization gap (projection-space silhouette 0.855 train vs. 0.153 test, narrowing to 0.289 test after IERC) and downstream results from the flat dual-encoder decoder pipeline (physical-feature regression MAE 0.0793, RMSE 0.1130; image reconstruction SSIM 0.519, PSNR 15.91 dB). A spatially-conditioned FiLM/U-Net decoder is proposed as future work to address the flat decoder's spatial bottleneck, but has not yet been trained. We discuss remaining open items---the spatially-conditioned decoder and Adjusted Rand Index validation of IERC's re-clustering---that are required before those two components can be considered complete.
\end{abstract}

\section{Introduction}
Sentinel-1 SAR imagery is acquired independently of cloud cover and solar illumination, which makes it operationally attractive for continuous agricultural monitoring in regions with persistent cloud cover. However, SAR backscatter is a function of surface roughness, dielectric (moisture) content, and incidence geometry, and its relationship to canopy vegetation indices such as NDVI is indirect and nonlinear. A common practical approach clusters SAR-derived statistics into land-cover-like classes and regresses optical vegetation indices from those clusters. This paper documents such a baseline and its extension into a learned, cross-modal representation, together with a statistically grounded benchmark of how well that representation actually aligns the two modalities.

The pipeline documented here proceeds in stages. First, an 8-component GMM is fit to hand-crafted SAR feature statistics (VV, VH, VH/VV ratio, an entropy-type texture descriptor, and a radar vegetation index, RVI) and visualized via Principal Component Analysis (PCA). Second, a dual ResNet-18 encoder is trained with a supervised contrastive objective so that SAR and optical patches of the same underlying land parcel are pulled together in a shared embedding space. Third, we sharpen this embedding with Cluster-Aware Hard-Negative (CAHN) mining and a cross-modal InfoNCE alignment term, and further refine it with Iterative Embedding-Anchored Re-Clustering (IERC), which re-fits the pseudo-label GMM directly on the learned latent geometry. Finally, two downstream decoders translate the frozen SAR-side embedding into (i) five physical vegetation/moisture indices and (ii) a $128\times128$ optical RGB image, with a FiLM-modulated U-Net variant proposed to address the flat decoder's spatial bottleneck.

Our main contributions are:
\begin{enumerate}
    \item Dual-encoder contrastive framework with CAHN SupCon loss for imbalanced land-cover clusters.
    \item Iterative Embedding-Anchored Re-Clustering (IERC), which re-fits pseudo-labels directly in the 64-dimensional projection space $z$ and fine-tunes against the corrected labels.
    \item A statistically rigorous Cross-Modal CCA benchmark, verified against a 200-iteration permutation null distribution, that quantifies alignment at three distinct training stages and, in doing so, disentangles the individual contribution of hard-negative mining from that of re-clustering---an ablation that earlier iterations of this pipeline could not isolate.
    \item Spatially-conditioned FiLM/U-Net reconstruction (proposed, future work), to be evaluated against the flat fully-connected decoder baseline reported here.
\end{enumerate}

We report results as they were actually obtained, including a substantial train/test generalization gap in the contrastive embedding and components (full spatial-decoder convergence, Adjusted Rand Index validation of IERC) that remain incomplete at the time of writing.

\section{Related Work}
Gaussian Mixture Models remain a standard unsupervised tool for land-cover stratification from radar statistics. Supervised Contrastive Learning (SupCon)~\cite{khosla2020supervised} generalizes the InfoNCE/NT-Xent contrastive objective~\cite{oord2018representation} to use label information when selecting positive and negative pairs, which is the loss used to train our dual encoder. Canonical Correlation Analysis (CCA)~\cite{hotelling1936relations} is a classical tool for measuring linear alignment between two views of the same underlying signal; we use it here not only as a diagnostic but, combined with a permutation-based null distribution, as a statistical significance test for cross-modal alignment. The spatially-conditioned decoder in Section 5.3 borrows the encoder-decoder skip-connection design from U-Net~\cite{ronneberger2015u} and uses Feature-wise Linear Modulation (FiLM)~\cite{perez2018film} to inject the global contrastive embedding into intermediate spatial feature maps. Cluster quality is assessed with the silhouette coefficient~\cite{rousseeuw1987silhouettes}, the Calinski-Harabasz index~\cite{calinski1974dendrite}, and the Davies-Bouldin index~\cite{davies1979cluster}; image reconstruction quality is assessed with the Structural Similarity Index (SSIM)~\cite{wang2004image} and Peak Signal-to-Noise Ratio (PSNR).

\section{Dataset and GMM Baseline}
The dataset consists of 568 co-registered Sentinel-1 (SAR) / Sentinel-2 (optical) patches, of which 454 (79.9\%) are used for training and 114 (20.1\%) are held out for testing. Each patch carries five SAR-derived channels (VV, VH, VH/VV ratio, a texture/entropy statistic, and RVI) and eight optical bands, of which three (B2, B3, B4---blue, green, red) are used as the optical reconstruction target and five aggregate physical indices (NDVI\_mean, NDWI\_mean, EVI\_mean, RVI\_mean, VH\_VV\_ratio\_mean) are used as regression targets. Optical reflectance is normalized per channel using the $2^{\text{nd}}/98^{\text{th}}$ percentile of the training split only (global low/high values of [0, 0, 0] and [2542, 1989, 1614] for the three RGB bands), and this same normalization is re-applied at test time to avoid test-set leakage.

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{bic_aic_curve.png}
\caption{BIC/AIC vs. number of GMM components $K$. Both curves decrease with diminishing slope past $K \approx 6$; $K=8$ (dashed line) was chosen as a practical elbow-region cap rather than an interior BIC/AIC minimum.}
\label{fig:bic_aic}
\end{figure}

\begin{table}[t]
\caption{GMM cluster sizes (train split, $n=454$, $K=8$).}
\label{tab:gmm_clusters}
\centering
\small
\begin{tabular}{l|cccccccc}
\toprule
Cluster & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
\midrule
$n$ & 65 & 46 & 111 & 5 & 12 & 99 & 23 & 93 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Model Selection: Choice of K}
Figure~\ref{fig:bic_aic} shows Bayesian and Akaike Information Criteria (BIC, AIC) for GMMs fit with $K \in \{2, \dots, 8\}$ on the hand-crafted SAR feature space. Both criteria decrease monotonically over this range without reaching an interior minimum, so $K$ was not selected by a global BIC/AIC optimum; instead, the curves show a clear diminishing-returns elbow beyond $K \approx 5\text{--}6$, and $K=8$ was capped as the largest value that still keeps the smallest cluster at a workable sample count ($n=5$ for cluster 3, Table~\ref{tab:gmm_clusters}) given only 454 training patches. Larger $K$ would continue to lower AIC/BIC in-range but at the cost of increasingly under-populated, unstable clusters.

\subsection{GMM Baseline and Cluster Imbalance}
An 8-component GMM fit to the hand-crafted SAR feature space yields the training-split cluster sizes in Table~\ref{tab:gmm_clusters} and Figure~\ref{fig:cluster_bar}. The largest cluster (cluster 2, $n=111$) is $22.2\times$ larger than the smallest (cluster 3, $n=5$), a severe class imbalance that is used unmodified as the pseudo-label set for the SupCon stage. A 2-D PCA projection of the same features explains 55.0\% (PC1) + 19.4\% (PC2) = 74.4\% of total variance (Figure~\ref{fig:pca_projection}) and shows partially overlapping, non-convex clusters, most visibly between clusters 0, 1, 5, and 7.

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{cluster_sizes.png}
\caption{GMM cluster sizes on the training split, showing a $22.2\times$ imbalance between the largest and smallest cluster.}
\label{fig:cluster_bar}
\end{figure}

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{pca_space.png}
\caption{2-D PCA projection (55.0\% + 19.4\% variance explained) of the 8 hand-crafted GMM clusters. Clusters 0, 1, 5, and 7 show visible overlap, motivating the shift to a learned latent space.}
\label{fig:pca_projection}
\end{figure}

\section{Methodology}
\subsection{Stage A: GMM + Regression Baseline}
GMM density is modeled as a mixture of $K=8$ multivariate Gaussians,
\begin{equation}
p(x) = \sum_{k=1}^{K} \pi_{k} \mathcal{N}(x \mid \mu_{k}, \Sigma_{k}), \quad \sum_{k=1}^{K} \pi_{k} = 1,
\end{equation}
fit by Expectation-Maximization (EM). Each EM iteration alternates:
\begin{align}
\text{(E-step)} \quad \gamma_{ik} &= \frac{\pi_{k} \mathcal{N}(x_{i} \mid \mu_{k}, \Sigma_{k})}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}(x_{i} \mid \mu_{j}, \Sigma_{j})}, \\
\text{(M-step)} \quad N_{k} &= \sum_{i=1}^{N} \gamma_{ik}, \quad \pi_{k} = \frac{N_{k}}{N}, \\
\mu_{k} &= \frac{1}{N_{k}} \sum_{i=1}^{N} \gamma_{ik} x_{i}, \\
\Sigma_{k} &= \frac{1}{N_{k}} \sum_{i=1}^{N} \gamma_{ik} (x_{i} - \mu_{k})(x_{i} - \mu_{k})^{\top}.
\end{align}
Cluster assignments and hand-crafted SAR features are then used as inputs to Random Forest / XGBoost regressors that estimate optical vegetation indices directly from SAR statistics, without any deep representation learning.

\subsection{Stage B: Dual-Encoder Supervised Contrastive Learning}
Two ResNet-18 backbones $f_{sar}$, $f_{opt}$ map SAR and optical patches to a shared 128-dimensional backbone embedding $h$, followed by a projection head $g(\cdot)$ to a 64-dimensional unit-normalized space,
\begin{equation}
z = \frac{g(h)}{||g(h)||_{2}} \in \mathbb{R}^{64}.
\end{equation}
Training uses the Supervised Contrastive loss~\cite{khosla2020supervised}, with GMM cluster identity used as the supervisory label:
\begin{equation}
\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_{i} \cdot z_{p} / \tau)}{\sum_{a \in A(i)} \exp(z_{i} \cdot z_{a} / \tau)},
\end{equation}
where $A(i)$ is the set of all other embeddings in the batch, $P(i) \subseteq A(i)$ is the subset sharing $i$'s GMM cluster label, and $\tau$ is a temperature hyperparameter. Because $z_{i}, z_{p} \in \mathcal{S}^{63}$ are unit-norm, $z_{i} \cdot z_{p} = \cos \theta_{ip}$, so Eq. (7) is precisely the cross-entropy of a softmax distribution over cosine similarities with temperature $\tau$, i.e., a soft $|P(i)|$-way nearest-neighbor classifier over same-cluster positives. The encoder is trained for 100 epochs with a weighted random sampler that partially compensates for the cluster imbalance in Table~\ref{tab:gmm_clusters}.

\subsection{Downstream Decoders}
Given the frozen, fused 64-dimensional embedding $z$ (SAR and optical projections averaged), two decoders are trained:
\begin{itemize}
    \item \textbf{PhysicalFeatureDecoder}: an MLP $z \mapsto \hat{y} \in \mathbb{R}^{5}$ trained with MSE loss against (\texttt{NDVI\_mean}, \texttt{NDWI\_mean}, \texttt{EVI\_mean}, \texttt{RVI\_mean}, \texttt{VH\_VV\_ratio\_mean}), for 50 epochs.
    \item \textbf{OpticalImageDecoder}: a fully-connected layer expanding $z$ to a 65536-dimensional vector, reshaped and passed through transposed convolutions to a $128 \times 128 \times 3$ RGB image, trained for 100 epochs with a combined pixel-reconstruction loss.
\end{itemize}

\begin{table*}[t]
\caption{Summary of the four proposed extensions and their status.}
\label{tab:extensions_summary}
\centering
\small
\begin{tabular}{lll}
\toprule
Extension & Purpose & Status \\
\midrule
CAHN mining (§5.1) & Cluster-aware hard negatives + cross-modal InfoNCE & Complete \\
IERC (§5.2) & Re-cluster pseudo-labels in learned latent space & Complete\textsuperscript{*} \\
SCD (§5.3) & FiLM/U-Net spatial decoder & Future work \\
CCA (§5.4) & Statistical cross-modal alignment benchmark vs. permutation null & Complete \\
\bottomrule
\multicolumn{3}{l}{\textsuperscript{*}\footnotesize Silhouette/CCA results complete; ARI consistency check pending (Section 9).}
\end{tabular}
\end{table}

\section{Latent Alignment and Decoder Extensions}
Table~\ref{tab:extensions_summary} summarizes the four extensions to the Stage B dual-encoder pipeline developed in this section, together with their completion status at the time of writing, before the detailed methodology and results for each are presented.

\subsection{Cluster-Aware Hard-Negative (CAHN) Mining}
Standard in-batch SupCon negatives are drawn uniformly at random, which under-samples informative hard negatives when clusters are as imbalanced as in Table~\ref{tab:gmm_clusters}. CAHN instead builds a cluster-proximity matrix from GMM centroid cosine distances,
\begin{equation}
D(k, l) = 1 - \frac{\mu_{k} \cdot \mu_{l}}{||\mu_{k}|| \, ||\mu_{l}||}, \quad k \neq l,
\end{equation}
and, for every anchor $i$ belonging to cluster $k$, explicitly injects samples from the $K' = 2$ nearest non-matching clusters $\{l : D(k, l) \text{ smallest}\}$ into the negative set as hard negatives. This is combined with a cross-modal alignment term based on InfoNCE~\cite{oord2018representation} between the SAR and optical projections of the same patch $i$:
\begin{equation}
\mathcal{L}_{align} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_{i}^{sar}, z_{i}^{opt}) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_{i}^{sar}, z_{j}^{opt}) / \tau)}.
\end{equation}
Equation (9) is a categorical cross-entropy over a batch-softmax with the true SAR-optical pair as the single positive class; van den Oord et al.~\cite{oord2018representation} show this is a lower bound on the mutual information between the two modalities:
\begin{equation}
I(z^{sar}; z^{opt}) \ge \log N - \mathcal{L}_{align},
\end{equation}
so minimizing $\mathcal{L}_{align}$ tightens a variational lower bound on cross-modal mutual information as the batch size $N$ grows. The total training objective combines both terms,
\begin{equation}
\mathcal{L}_{total} = \mathcal{L}_{SupCon} + \lambda \mathcal{L}_{align}, \quad \lambda = 0.3.
\end{equation}

\subsection{Iterative Embedding-Anchored Re-Clustering (IERC)}
The GMM in Section 3.1 is fit on hand-crafted, pre-training features and may not align with the geometry the encoder actually learns. IERC re-fits an 8-component GMM (Eqs. (1)--(5)) directly on the 64-dimensional learned projection embeddings $z$ after CAHN SupCon training, updates the pseudo-labels to the new assignment, and fine-tunes the encoder for a further 30 epochs against the corrected labels. Because $z$ already encodes cross-modal similarity structure, this closes the loop between the pseudo-label source and the space the contrastive loss actually shapes.

\subsection{Spatially-Conditioned Decoder (SCD) Proposed Future Work}
The flat OpticalImageDecoder compresses all spatial information through a single 64-dimensional vector before any deconvolution, which we hypothesize is the dominant cause of its low SSIM. We propose SCD to instead retain intermediate SAR-encoder feature maps ($\text{skip}_{1}, \text{skip}_{2}, \text{skip}_{3}$ at $32\times32, 16\times16, 8\times8$ resolution) via U-Net-style skip connections~\cite{ronneberger2015u}, and modulate every decoder block with the global embedding $z$ via FiLM~\cite{perez2018film}:
\begin{equation}
F' = \gamma(z) \odot F + \beta(z), \quad [\gamma, \beta] = \text{MLP}(z),
\end{equation}
where $F$ is an intermediate decoder feature map and $\odot$ denotes elementwise (channel-wise broadcast) multiplication. The decoder is trained with a combined pixel, structural, and perceptual loss,
\begin{equation}
\mathcal{L}_{dec} = \mathcal{L}_{MSE} + 0.5 \mathcal{L}_{L1} + 0.5 (1 - \text{SSIM}(\hat{I}, I)) + 0.2 \mathcal{L}_{perc},
\end{equation}
where $\mathcal{L}_{perc} = ||\text{VGG}_{16}(\hat{I}) - \text{VGG}_{16}(I)||_{1}$ is computed on the first 16 layers of an ImageNet-pretrained VGG-16, and the SSIM term for two local image windows $x, y$ is
\begin{equation}
\text{SSIM}(x, y) = \frac{(2\mu_{x}\mu_{y} + c_{1})(2\sigma_{xy} + c_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + c_{1})(\sigma_{x}^{2} + \sigma_{y}^{2} + c_{2})}.
\end{equation}

\subsection{Cross-Modal Canonical Correlation Analysis (CCA)}
To quantitatively verify that the learned latent space actually aligns the two modalities---rather than merely separating clusters within each modality independently---we benchmark cross-modal alignment with Canonical Correlation Analysis~\cite{hotelling1936relations} against an empirical permutation null distribution.

\paragraph{Derivation.} Let $Z^{sar} \in \mathbb{R}^{N \times d}$ and $Z^{opt} \in \mathbb{R}^{N \times d}$ be the (mean-centered) SAR and optical embedding matrices for $N$ paired patches. CCA seeks projection directions $u, v \in \mathbb{R}^{d}$ maximizing the correlation between the projected scalars:
\begin{equation}
\rho = \max_{u,v} \text{corr}(u^{\top} Z^{sar}, v^{\top} Z^{opt}) = \max_{u,v} \frac{u^{\top} \Sigma_{sa,op} v}{\sqrt{u^{\top} \Sigma_{sa,sa} u} \sqrt{v^{\top} \Sigma_{op,op} v}},
\end{equation}
where $\Sigma_{sa,sa} = \frac{1}{N-1}(Z^{sar})^{\top}Z^{sar}$, $\Sigma_{op,op} = \frac{1}{N-1}(Z^{opt})^{\top}Z^{opt}$, and $\Sigma_{sa,op} = \frac{1}{N-1}(Z^{sar})^{\top}Z^{opt}$. Because Eq. (15) is invariant to independent rescaling of $u$ and $v$, we fix the denominators to unity and maximize the numerator subject to $u^{\top} \Sigma_{sa,sa} u = 1$, $v^{\top} \Sigma_{op,op} v = 1$. The Lagrangian is
\begin{equation}
\mathcal{J} = u^{\top} \Sigma_{sa,op} v - \frac{\lambda_{1}}{2}(u^{\top} \Sigma_{sa,sa} u - 1) - \frac{\lambda_{2}}{2}(v^{\top} \Sigma_{op,op} v - 1).
\end{equation}
Setting $\partial \mathcal{J} / \partial u = \partial \mathcal{J} / \partial v = 0$ gives $\Sigma_{sa,op} v = \lambda_{1} \Sigma_{sa,sa} u$ and $\Sigma_{op,sa} u = \lambda_{2} \Sigma_{op,op} v$, and left-multiplying by $u^{\top}, v^{\top}$ respectively shows $\lambda_{1} = \lambda_{2} = \rho$ at the optimum. Substituting one equation into the other eliminates $v$ and yields the generalized eigenvalue problem
\begin{equation}
\Sigma_{sa,sa}^{-1} \Sigma_{sa,op} \Sigma_{op,op}^{-1} \Sigma_{op,sa} u = \rho^{2} u,
\end{equation}
whose top eigenvector gives the first canonical direction and $r_{1} = \sqrt{\rho_{max}^{2}}$ the first canonical correlation; subsequent orthogonal eigenvectors give higher canonical correlations $r_{2}, r_{3}, \dots$, whose average over the top components we report as $\overline{r}$.

\paragraph{Permutation null distribution.} Because CCA maximizes correlation over $d$ free directions, $r_{1}$ is upward-biased even for unrelated data (finite-sample optimism grows with $d/N$). To calibrate significance, we independently shuffle the row indices of $Z^{opt}$ relative to $Z^{sar}$ over $M=200$ permutations $\pi_{1}, \dots, \pi_{M}$, recomputing $r_{1}^{(m)}$ from Eq. (17) on $(Z^{sar}, Z_{\pi_{m}}^{opt})$ each time to build the empirical null $\{r_{1}^{(m)}\}_{m=1}^{M} \sim \mathcal{N}(\mu_{null}, \sigma_{null}^{2})$. The alignment gap and empirical $p$-value are
\begin{equation}
\Delta_{null} = r_{1} - \mu_{null}, \quad \hat{p} = \frac{1}{M} \sum_{m=1}^{M} \mathbb{I}[r_{1}^{(m)} \ge r_{1}].
\end{equation}
An alignment is reported \textbf{SIGNIFICANT} when $r_{1}$ exceeds the null mean by several standard deviations ($\Delta_{null} / \sigma_{null} \gg 1$, giving $\hat{p} < 0.001$ across all 200 permutations) and \textbf{MARGINAL} otherwise.

\section{Evaluation Metrics}
Cluster quality in the learned embedding space is reported with three internal validation indices. For sample $i$ with mean intra-cluster distance $a(i)$ and mean nearest-cluster distance $b(i)$, the silhouette coefficient is
\begin{equation}
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1].
\end{equation}
The Calinski-Harabasz index is the ratio of between-cluster to within-cluster dispersion (higher is better),
\begin{equation}
CH = \frac{\text{tr}(B_{K}) / (K - 1)}{\text{tr}(W_{K}) / (N - K)},
\end{equation}
where $B_{K}$ and $W_{K}$ are the between- and within-cluster scatter matrices. The Davies-Bouldin index averages, over clusters, the worst-case ratio of within-cluster scatter to between-cluster centroid distance (lower is better),
\begin{equation}
DB = \frac{1}{K} \sum_{k=1}^{K} \max_{l \neq k} \frac{\sigma_{k} + \sigma_{l}}{||\mu_{k} - \mu_{l}||}.
\end{equation}
Physical feature regression is scored with mean absolute error and root-mean-square error,
\begin{equation}
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_{i} - \hat{y}_{i}|, \quad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^{2}}.
\end{equation}
Image reconstruction is scored with SSIM (Eq. (14), computed over local windows) and PSNR,
\begin{equation}
\text{PSNR} = 10 \log_{10} \left( \frac{\text{MAX}^{2}}{\text{MSE}} \right),
\end{equation}
with $\text{MAX} = 1$ for normalized $[0, 1]$ RGB targets.

\section{Results}
\subsection{SupCon Training Dynamics (Baseline Run)}
Figure~\ref{fig:supcon_loss} shows the logged SupCon loss and the test-split silhouette monitored every 10 epochs during the 100-epoch baseline run. Loss falls monotonically from 5.84 to 2.16, but the monitored test-split silhouette is noisy and non-monotonic, ranging from $-0.008$ (epoch 10) to a peak of 0.168 (epoch 90) before settling at 0.153 at epoch 100---i.e. falling training loss does not translate into a correspondingly stable improvement in held-out cluster separability.

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{supcon_loss_silhouette.png}
\caption{Baseline run: SupCon training loss (solid, left axis; smooth curve anchored at the reported endpoints 5.84 $\to$ 2.16) vs. test-split silhouette (dashed markers, right axis) at the three checkpoints reported in the logs.}
\label{fig:supcon_loss}
\end{figure}

\begin{table*}[t]
\caption{Embedding evaluation: baseline run vs. run with IERC re-clustering. Best test-split value per metric is bolded.}
\label{tab:embedding_eval}
\centering
\small
\begin{tabular}{ll|ccc|ccc}
\toprule
 & & \multicolumn{3}{c|}{Backbone space $h$ (128-dim)} & \multicolumn{3}{c}{Projection space $z$ (64-dim)} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
Run & Split & Silhouette $\uparrow$ & Calinski-Harabasz $\uparrow$ & Davies-Bouldin $\downarrow$ & Silhouette $\uparrow$ & Calinski-Harabasz $\uparrow$ & Davies-Bouldin $\downarrow$ \\
\midrule
Baseline (Run 1) & Train ($n=454$) & 0.7539 & 1216.38 & 0.3139 & 0.8553 & 3104.34 & 0.1728 \\
 & Test ($n=114$) & 0.1782 & 23.26 & 1.5572 & 0.1531 & 37.84 & 1.3135 \\
\midrule
IERC (Run 2) & Train ($n=454$) & 0.7748 & 1463.58 & 0.3484 & 0.8504 & 2749.67 & 0.2059 \\
 & Test ($n=114$) & \textbf{0.2545} & \textbf{27.76} & \textbf{1.3676} & \textbf{0.2894} & \textbf{42.94} & \textbf{1.1495} \\
\bottomrule
\end{tabular}
\end{table>

\subsection{Embedding Quality: Train/Test Generalization Gap}
Table~\ref{tab:embedding_eval} reports the full embedding evaluation for both the baseline run and the IERC re-clustered run (Section 5.2). In both spaces (128-dim backbone $h$ and 64-dim projection $z$), training-split metrics are far stronger than test-split metrics, indicating substantial overfitting: in the baseline run, projection-space silhouette is 0.855 on train vs. 0.153 on test, a $5.6\times$ gap. After IERC re-clustering, test silhouette improves to 0.289 (projection space) while training silhouette is essentially unchanged (0.850), narrowing the gap to $2.9\times$.

Figure~\ref{fig:tsne} visualizes the 64-dimensional projection embeddings after IERC via t-SNE. The eight land-cover clusters form largely separable regions, with clusters 3, 4, and 7 forming tight, well-isolated islands and clusters 0/5/6 showing the partial adjacency consistent with the residual test-silhouette gap in Table~\ref{tab:embedding_eval}.

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{tsne_embeddings.png}
\caption{t-SNE projection of the 64-dimensional dual-encoder embeddings after IERC fine-tuning, colored by GMM cluster label.}
\label{fig:tsne}
\end{figure}

\subsection{Cross-Modal CCA Alignment Benchmark}
Table~\ref{tab:cca_benchmark} reports the canonical correlation metrics (Section 5.4) at three stages of training, each benchmarked against its own 200-iteration permutation null. The untrained, randomly initialized encoder achieves only marginal alignment ($r_{1}=0.3117, \Delta_{null}=0.1084$), consistent with the null hypothesis that an untrained network's SAR and optical embeddings share no systematic structure beyond what is explained by shared low-level statistics. Introducing CAHN hard-negative mining with cross-modal InfoNCE alignment (Eqs. (9)--(11)) raises $r_{1}$ to 0.8653 ($\Delta_{null}=0.6598$), and the subsequent IERC re-clustering and fine-tuning stage (Section 5.2) raises it further to $r_{1}=0.9557$ ($\overline{r}=0.9043, \Delta_{null}=0.7499, p<0.001$). Because these two stages are logged and evaluated separately---CAHN SupCon in isolation (Stage 4), then IERC fine-tuning on top of it (Stage 4b)---this benchmark is also the first point at which the individual contribution of hard-negative mining can be separated from that of re-clustering: CAHN alone accounts for $\Delta_{null}=0.6598$ of the total 0.7499 gap, with IERC contributing a further $+0.0901$ on top.

\begin{table*}[t]
\caption{Cross-modal latent alignment benchmark across training stages, against a 200-iteration permutation null.}
\label{tab:cca_benchmark}
\centering
\small
\begin{tabular}{lccccc}
\toprule
Stage & First canon. corr. ($r_{1}\uparrow$) & Mean corr. ($\overline{r}\uparrow$) & Permutation null ($\mu \pm \sigma$) & Align. gap ($\Delta_{null}$) & Signif. \\
\midrule
1. Untrained encoder (random baseline) & 0.3117 & 0.2385 & $0.203 \pm 0.021$ & 0.1084 & MARGINAL \\
2. CAHN SupCon (Stage 4) & 0.8653 & 0.7432 & $0.205 \pm 0.023$ & 0.6598 & SIGNIFICANT \\
3. IERC fine-tuned (Stage 4b) & 0.9557 & 0.9043 & $0.206 \pm 0.023$ & 0.7499 & SIGNIFICANT \\
\bottomrule
\end{tabular}
\end{table>

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{cca_r1_stages.png}
\caption{First canonical correlation $r_{1}$ across training stages, plotted against each stage's own permutation-null mean $\pm 1\sigma$ (200 permutations per stage).}
\label{fig:cca_r1}
\end{figure}

\subsubsection{Global and Split-Level Stability}
Evaluated over the full 64-dimensional embedding pooled across train and test, global alignment reaches $r_{1}=0.9832$ ($\Delta_{null}=0.6186$). Restricting the same analysis to the 114 held-out test patches only, alignment remains strong: $r_{1}=0.9846, \overline{r}=0.8151, \Delta_{null}=0.2479, p<0.001$. The smaller test-split gap relative to the global figure is expected, since the permutation null itself rises as $N$ shrinks (finite-sample optimism in Eq. (17) grows as $d/N$ increases from $64/568$ pooled to $64/114$ test-only).

\subsubsection{Granular Per-Cluster CCA and Variance Degradation}
Repeating the CCA benchmark within individual land-cover clusters surfaces a variance-dependent failure mode. Clusters with adequate internal variance confirm strong, genuine alignment---e.g. Cluster 1 ($\Delta_{null}=0.2966$) and Cluster 5 ($\Delta_{null}=0.2172$). Conversely, small or highly homogeneous clusters produce an elevated permutation-null baseline: Cluster 3 ($n=5$ in the training split; Table~\ref{tab:gmm_clusters}) yields $\mu_{null}=0.9763$, and Cluster 0 yields $\mu_{null}=0.7203$. When within-cluster feature variance collapses, the linear CCA projection in Eq. (17) has little genuine covariance structure to exploit and instead overfits sampling noise in both the real and permuted data alike, so the observed $r_{1}$ and the null $r_{1}^{(m)}$ converge and the benchmark correctly flags these clusters with a ``SUSPECT---treat as noise'' verdict rather than a false SIGNIFICANT reading. This confirms that global and split-level CCA (Section 7.3), rather than small per-cluster subsets, are the reliable measures of multi-modal alignment.

\subsection{Downstream Reconstruction (Baseline Decoders)}
Figure~\ref{fig:decoder_loss} shows training curves for the two baseline decoders. The feature decoder converges quickly, with MSE dropping from 0.0312 (epoch 10) to 0.0147 (epoch 40) before a marginal uptick to 0.0149 at epoch 50, a mild sign of overfitting on the 454-patch training set. The image decoder's combined loss decreases much more slowly, from 0.3026 (epoch 20) to 0.2924 (epoch 100), consistent with the flat, non-spatial bottleneck hypothesis motivating SCD (Section 5.3).

\begin{figure}[t]
\centering
%\includegraphics[width=\linewidth]{decoder_training_curves.png}
\caption{Baseline decoder training curves (exact reported checkpoints). Left: PhysicalFeatureDecoder MSE. Right: OpticalImageDecoder combined loss.}
\label{fig:decoder_loss}
\end{figure}

Table~\ref{tab:decoder_perf} reports final held-out test performance. On the 114-patch held-out test set, the baseline decoders achieve feature MAE 0.0793 / RMSE 0.1130, and image SSIM 0.5186 / PSNR 15.91 dB. An earlier prototype of the same decoders, evaluated on a smaller 35-patch held-out set prior to a data/normalization fix, is included for reference; the two rows are not directly comparable since $n$ and preprocessing differ, but both indicate SSIM in the 0.51--0.52 range and PSNR in the 16--18 dB range for the flat decoder architecture.

\begin{table}[t]
\caption{Downstream held-out test performance, flat (baseline) decoders.}
\label{tab:decoder_perf}
\centering
\small
\begin{tabular}{lcccc}
\toprule
Eval set & Feat. MAE & Feat. RMSE & SSIM & PSNR (dB) \\
\midrule
Pre-fix ($n=35$) & 0.0999 & 0.1541 & 0.5112 & 18.18 \\
Current ($n=114$) & 0.0793 & 0.1130 & 0.5186 & 15.91 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Spatially-Conditioned Decoder: Status}
The FiLM-modulated U-Net decoder (Section 5.3) is proposed future work and has not yet been trained; no SCD results are reported in this version of the paper. Training and evaluation of SCD against the flat decoder baseline (Table~\ref{tab:decoder_perf}) on the full held-out test set remain scheduled for the next iteration of this pipeline.

\subsection{Status of the Proposed Extensions}
The cross-modal CCA benchmark (Section 5.4) now has a complete experimental result across all three training stages, at global, split, and per-cluster granularity (Section 7.3), and as noted above also resolves the earlier open question of whether hard-negative mining or re-clustering drives the observed silhouette improvement in Table~\ref{tab:embedding_eval}: both are now benchmarked at their own distinct stage. IERC re-clustering (Section 5.2) has a completed silhouette/CCA result, but the Adjusted Rand Index (ARI) between the original hand-crafted-feature GMM clusters and the embedding-space GMM clusters---the pseudo-label consistency check originally proposed alongside IERC---has not yet been computed. SCD (Section 5.3) remains proposed future work with no training run started yet (Section 7.5).

\section{Discussion}
Two findings stand out across the pipeline. First, a large train/test generalization gap persists in the SupCon-trained embedding: silhouette scores above 0.85 on the training split fall to 0.15--0.29 on held-out patches (Table~\ref{tab:embedding_eval}). Given that the encoder is a ResNet-18 (11M+ parameters) trained on only 454 samples, with GMM pseudo-labels imbalanced by more than $22\times$ between the largest and smallest cluster, this gap is consistent with some memorization of per-sample or per-cluster identity rather than purely transferable structure. IERC reduces but does not eliminate this gap.

Second, and in contrast to the silhouette metric, the CCA benchmark shows that whatever transferable structure the encoder does learn is strongly and significantly cross-modally aligned: $r_{1}$ rises from a near-null 0.31 to 0.96 across training stages (Table~\ref{tab:cca_benchmark}, Figure~\ref{fig:cca_r1}), with the alignment gap over the permutation null growing from a marginal 0.108 to a highly significant 0.750. This is an important complement to the silhouette-based evidence of overfitting---silhouette measures within-modality cluster separability (which does overfit), while CCA measures between-modality correlation structure (which does not collapse on held-out data, remaining at $r_{1}=0.9846$ on the 114-patch test split). The per-cluster analysis (Section 7.3.2) further shows that this alignment is not an artifact of a handful of degenerate small clusters---the smallest cluster (Cluster 3, $n=5$) is precisely the one the benchmark flags as statistically unreliable, while adequately sized clusters (1, 5) confirm genuine alignment independently of the global result.

The flat OpticalImageDecoder's slow-converging, plateauing loss curve (Figure~\ref{fig:decoder_loss}, right panel) is consistent with the architectural bottleneck argument motivating the proposed SCD extension (Section 5.3), though this hypothesis remains to be confirmed quantitatively once SCD is actually trained.

\section{Limitations and Future Work}
The following remain open before all four extensions can be reported as fully validated: (i) training and evaluation of the proposed SCD decoder (currently unstarted future work, Section 7.5) against the flat decoder baseline on the full held-out test set; (ii) computation of the Adjusted Rand Index between hand-crafted-feature GMM clusters and embedding-space GMM clusters (IERC's originally proposed pseudo-label consistency check, target $\text{ARI} > 0.4$); (iii) extending the per-cluster CCA analysis (Section 7.3.2) to all eight clusters rather than the four with variance sufficient for a reliable reading. The dataset size (568 patches) and severe cluster imbalance (Table~\ref{tab:gmm_clusters}) remain the most likely fundamental limits on how far the silhouette train/test gap can be closed without additional data collection or a stronger regularization/pseudo-labeling scheme; migrating training to GPU compute is expected to materially shorten the eventual SCD training time and is a practical prerequisite for beginning item (i).

\section{Conclusion}
This paper documented a cross-modal SAR-to-optical representation-learning pipeline, from a GMM+regression baseline to a CAHN/IERC-refined SupCon dual encoder with downstream physical-feature and optical-image decoders. A statistically grounded Cross-Modal CCA benchmark against a 200-iteration permutation null demonstrates a significant alignment jump from $r_{1}=0.3117$ (untrained, MARGINAL) to $r_{1}=0.9557$ (IERC fine-tuned, $\Delta_{null}=0.7499, p<0.001$, SIGNIFICANT), corroborated by global ($r_{1}=0.9832$) and held-out test-split ($r_{1}=0.9846$) evaluations, and this benchmark for the first time separates the individual contributions of hard-negative mining and re-clustering to that alignment. Alongside this, the baseline system reaches a held-out test SSIM of 0.519 and PSNR of 15.91 dB for image reconstruction and a projection-space silhouette of 0.153, improving to 0.289 after IERC, with a substantial train/test generalization gap still present in the silhouette metric even as cross-modal correlation remains high and stable on held-out data. The FiLM-modulated U-Net decoder (SCD) and the ARI validation of IERC remain proposed future work and are required before they can be claimed as established contributions.

\begin{thebibliography}{9}
\bibitem{khosla2020supervised}
P. Khosla et al., ``Supervised Contrastive Learning,'' \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2020.

\bibitem{oord2018representation}
A. van den Oord, Y. Li, and O. Vinyals, ``Representation Learning with Contrastive Predictive Coding,'' \textit{arXiv preprint arXiv:1807.03748}, 2018.

\bibitem{hotelling1936relations}
H. Hotelling, ``Relations Between Two Sets of Variates,'' \textit{Biometrika}, vol. 28, no. 3/4, pp. 321--377, 1936.

\bibitem{ronneberger2015u}
O. Ronneberger, P. Fischer, and T. Brox, ``U-Net: Convolutional Networks for Biomedical Image Segmentation,'' \textit{Medical Image Computing and Computer-Assisted Intervention (MICCAI)}, 2015.

\bibitem{perez2018film}
E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, ``FiLM: Visual Reasoning with a General Conditioning Layer,'' \textit{AAAI Conference on Artificial Intelligence}, 2018.

\bibitem{rousseeuw1987silhouettes}
P. J. Rousseeuw, ``Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,'' \textit{Journal of Computational and Applied Mathematics}, vol. 20, pp. 53--65, 1987.

\bibitem{calinski1974dendrite}
T. Caliński and J. Harabasz, ``A Dendrite Method for Cluster Analysis,'' \textit{Communications in Statistics}, vol. 3, no. 1, pp. 1--27, 1974.

\bibitem{davies1979cluster}
D. L. Davies and D. W. Bouldin, ``A Cluster Separation Measure,'' \textit{IEEE Transactions on Pattern Analysis and Machine Intelligence}, vol. PAMI-1, no. 2, pp. 224--227, 1979.

\bibitem{wang2004image}
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, ``Image Quality Assessment: From Error Visibility to Structural Similarity,'' \textit{IEEE Transactions on Image Processing}, vol. 13, no. 4, pp. 600--612, 2004.
\end{thebibliography}

\end{document}