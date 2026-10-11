const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  Table,
  TableRow,
  TableCell,
  WidthType,
  ShadingType
} = require("docx");

// Helper functions for consistent styling and document layout
const H1 = (text) =>
  new Paragraph({
    text,
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 150 }
  });

const H2 = (text) =>
  new Paragraph({
    text,
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 }
  });

const P = (text, opts = {}) =>
  new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 }
  });

const Bullet = (text) =>
  new Paragraph({
    text,
    bullet: { level: 0 },
    spacing: { after: 100 }
  });

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1600, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    children: [
      new Paragraph({
        children: [
          new TextRun({
            text: String(text),
            bold: !!opts.header,
            size: opts.size || 20
          })
        ]
      })
    ]
  });
}

function makeTable(headers, rows, widths) {
  const w = widths || headers.map(() => Math.floor(9000 / headers.length));
  const headerRow = new TableRow({
    children: headers.map((h, i) => cell(h, { header: true, width: w[i] }))
  });
  const bodyRows = rows.map(
    (r) =>
      new TableRow({
        children: r.map((v, i) => cell(v, { width: w[i] }))
      })
  );
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: w,
    rows: [headerRow, ...bodyRows]
  });
}

// Assemble full document content
const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({
          children: [
            new TextRun({
              text: "Cluster-Aware Cross-Modal Representation Learning for SAR–Optical Earth Observation:",
              bold: true,
              size: 30
            })
          ],
          spacing: { after: 60 }
        }),
        new Paragraph({
          children: [
            new TextRun({
              text: "A Comparative Study with CCA Validation and Physics-Grounded Failure Analysis",
              bold: true,
              size: 30
            })
          ],
          spacing: { after: 200 }
        }),
        P("Author: Mohammed Kaamraan Ahmed", { bold: true }),
        P("Institution: Indian Institute of Information Technology, Sri City (IIITS)", { bold: true }),
        P(
          "Status: Complete draft — representation learning, multi-seed validation, Canonical Correlation Analysis, and direct baseline comparison are all implemented and evaluated.",
          { italics: true }
        ),

        H1("Abstract"),
        P(
          "Optical satellite sensors provide detailed spectral information for land-cover monitoring but are frequently obscured by cloud cover. Active Synthetic Aperture Radar (SAR) penetrates atmospheric obstacles but produces backscatter signals structurally very different from optical reflectance. This work develops a dual-encoder representation-learning framework mapping co-registered Sentinel-1/Sentinel-2 patches, sampled across four physically diverse Indian agricultural regions, into a shared latent space trained with Supervised Contrastive Loss (SupCon)."
        ),
        P(
          "Across three independent random splits, downstream probes classify GMM land-cover labels from SAR embeddings at 68.6–71.4% mean accuracy against a 31.4% majority-class baseline; cross-modal retrieval achieves 87.6% \u00b1 1.3% Top-5 accuracy. A Canonical Correlation Analysis (CCA), validated against a permutation-test null baseline to rule out dimensionality-driven artifacts, confirms genuine shared variance between SAR and optical data overall (gap over null = 0.58 at the descriptor level, 0.62 at the embedding level) and identifies specific land-cover clusters where this correlation is statistically indistinguishable from noise."
        ),
        P(
          "A direct comparison against the original hand-crafted-descriptor RF/XGBoost baseline, evaluated on the identical held-out test patches, shows that representation learning does NOT uniformly outperform simpler methods: the learned embedding wins in 3 of 5 evaluable clusters and loses in 2, and even in aggregate, it achieves lower Mean Absolute Error (0.111 vs. 0.119) but worse R\u00b2 (0.62 vs. 0.69) than the simplest baseline. Critically, the clusters where representation learning fails (Cluster 0) and succeeds (Cluster 1) are independently confirmed by three separate analyses \u2014 CCA, downstream probing/retrieval, and direct regression comparison \u2014 converging on a physically coherent explanation: correlation weakens under dense-canopy backscatter saturation and strengthens in sparser, more SAR-transparent conditions. This thesis's core contribution is this conditional, cluster-aware account of when cross-modal representation learning helps and when it does not, rather than a single aggregate accuracy claim."
        ),

        H1("1. Problem Statement"),
        P(
          "Existing SAR-optical studies largely focus on maximizing prediction accuracy with new models, rarely asking how the cross-modal relationship changes across land-cover conditions or why representation learning helps in some regimes and not others. This thesis addresses that gap directly, using Canonical Correlation Analysis and a controlled baseline comparison \u2014 not just downstream task accuracy \u2014 to characterize where SAR-optical correlation is real and where it is not."
        ),

        H1("2. Dataset"),
        P(
          "172 co-registered SAR\u2013optical patches across four diverse Indian agricultural regions: Punjab (dense wheat/rice), Krishna-Godavari Delta (flooded paddy), Vidarbha (sparse/semi-arid cotton), Kerala backwaters (tropical wet, high cloud). Split 80/20 (137 train / 35 test), stratified by GMM cluster (K=7)."
        ),

        H1("3. Methodology"),
        Bullet(
          "Stages 1\u20133: Band math (NDVI, NDWI, EVI, RVI, VH/VV ratio, entropy proxy) + GMM pseudo-labeling."
        ),
        Bullet(
          "Stage 4: Dual ResNet-18 encoders (ImageNet-pretrained layer1\u20134, re-initialized conv1), SupCon training (\u03c4=0.07), 64-dim L2-normalized projection space."
        ),
        Bullet(
          "Stage 5\u20136: Embedding evaluation, downstream probing/retrieval on held-out test patches."
        ),
        Bullet(
          "Stage 8: Physical Feature Decoder (MLP) predicting 5 descriptors from frozen SAR embeddings, train-split only."
        ),
        Bullet(
          "NEW \u2014 Canonical Correlation Analysis: run at both the raw-descriptor level (SAR: VV, VH, VH/VV, RVI, entropy vs. Optical: B2-B8, NDVI, NDWI, EVI) and the learned-embedding level (separate SAR/optical projection vectors), overall and per cluster. Embedding-level CCA required PCA dimensionality reduction (64 dims down to ~n_samples/5) to avoid a known CCA failure mode where high dimensionality relative to sample size produces spuriously perfect correlations; every result is validated against a 200-iteration permutation-shuffled null baseline, and only reported as real when it exceeds that null by a meaningful margin."
        ),
        Bullet(
          "NEW \u2014 Baseline comparison: RF/XGBoost fit on the SAME train split and evaluated on the SAME held-out test split as the embedding pipeline, predicting NDVI from hand-crafted SAR descriptors, then compared directly against the NDVI-specific component of the Stage 8 decoder's output on the identical test patches."
        ),

        H1("4. Results"),
        H2("4.1 Multi-Seed Representation Learning (3 seeds: 42, 1, 7)"),
        makeTable(
          ["Metric", "Mean", "Std Dev"],
          [
            ["Best Probe Accuracy (RBF SVM)", "71.43%", "6.17"],
            ["Top-5 Retrieval Accuracy", "87.62%", "1.35"],
            ["Mean Reciprocal Rank", "0.72", "0.03"]
          ],
          [4000, 2500, 2500]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        P(
          "Baselines: random = 14.3% (7 classes), majority-class = 31.4%. All probe accuracies remain well above baseline across every seed."
        ),

        H2("4.2 Canonical Correlation Analysis"),
        P(
          "Every CCA result below is reported as the gap between the observed canonical correlation and a permutation-shuffled null baseline (200 shuffles) \u2014 a gap under ~0.15 is not distinguishable from a dimensionality/sample-size artifact and is marked SUSPECT rather than treated as a finding."
        ),
        makeTable(
          ["Group", "Descriptor-Level Gap", "Verdict", "Embedding-Level Gap", "Verdict"],
          [
            ["Overall", "0.579", "REAL", "0.619", "REAL"],
            ["Cluster 0", "0.092", "SUSPECT", "0.023", "SUSPECT"],
            ["Cluster 1", "0.297", "REAL", "0.235", "REAL"],
            ["Cluster 5", "0.217", "REAL", "0.080", "SUSPECT"],
            ["Cluster 6", "0.069", "SUSPECT", "0.200", "REAL"]
          ],
          [1800, 1800, 1400, 1800, 1400]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        P(
          "Overall SAR-optical correlation is confirmed real at both levels. Cluster 0 shows NO detectable correlation at either the raw-descriptor or learned-embedding level \u2014 the first of three independent confirmations of this cluster's failure (see Section 4.4). Cluster 1 shows real correlation at both levels \u2014 the first of three independent confirmations of this cluster's success. Clusters 5 and 6 show a real signal at only one of the two levels, a genuine and reported nuance rather than a forced clean pattern."
        ),

        H2("4.3 Baseline Comparison: Hand-Crafted Descriptors vs. Learned Embedding"),
        P(
          "Both methods predict NDVI_mean; both are fit on the identical 137-patch train split and evaluated on the identical 35-patch held-out test split."
        ),
        makeTable(
          ["Method", "R\u00b2", "RMSE", "MAE"],
          [
            ["Hand-crafted RF baseline", "0.685", "0.1684", "0.1187"],
            ["Hand-crafted XGBoost baseline", "0.656", "0.1759", "0.1229"],
            ["Learned SAR embedding decoder", "0.623", "0.1841", "0.1107"]
          ],
          [4000, 1500, 1750, 1750]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        P(
          "The embedding achieves lower MAE but worse R\u00b2/RMSE than the RF baseline \u2014 a genuine trade-off, not a clean win. This pattern (lower typical-case error, higher sensitivity to outlier patches) is consistent with a learned representation that generalizes well on common cases but is less robust than a simple baseline on harder ones."
        ),

        H2("4.4 Per-Cluster Baseline Comparison"),
        makeTable(
          ["Cluster", "n", "RF MAE", "XGB MAE", "Embedding MAE", "Winner"],
          [
            ["0", "6", "0.0689", "0.0878", "0.1039", "Baseline"],
            ["1", "11", "0.0893", "0.0805", "0.0664", "Embedding"],
            ["3", "3", "0.1949", "0.1988", "0.1426", "Embedding"],
            ["5", "10", "0.1141", "0.1179", "0.0693", "Embedding"],
            ["6", "4", "0.0925", "0.1123", "0.1488", "Baseline"]
          ],
          [1200, 700, 1500, 1500, 1900, 1700]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        P(
          "Representation learning wins in 3 of 5 evaluable clusters. Combined with Section 4.2's CCA results and the earlier probing/retrieval breakdown, Clusters 0 and 1 are now confirmed by THREE independent analyses: Cluster 0 consistently shows no cross-modal benefit (weakest CCA, weakest probing/retrieval, and now the baseline beats the embedding); Cluster 1 consistently shows strong cross-modal benefit across all three. Clusters 5 and 6 show partial agreement across methods \u2014 reported honestly as a more nuanced pattern rather than forced into the same clean story."
        ),

        H1("5. Discussion: A Cluster-Conditional, Physically Grounded Account"),
        P(
          "Cluster 0 is characterized by comparatively high NDVI/RVI/entropy-proxy values (dense vegetation). Cluster 1, the strongest cluster across every analysis, has the lowest values among sizable clusters (sparser canopy). This is consistent with SAR backscatter saturation: as canopy closes, radar backscatter becomes less sensitive to further biophysical variation, degrading its correlation with optical descriptors \u2014 and this degradation shows up consistently whether measured via CCA, downstream classification, or direct regression."
        ),
        P(
          "The region-level pattern reinforces this: Vidarbha (sparsest canopy) has the best reconstruction fidelity; Kerala backwaters (cloud-prone, high-moisture) has the worst feature MAE, consistent with moisture-driven backscatter noise and cloud-driven acquisition-date mismatch; Krishna-Godavari (flooded paddy) shows a split between strong classification-level retrieval and weak pixel-level reconstruction, consistent with a distinctive but structurally complex double-bounce scattering signature."
        ),
        P(
          "The central methodological lesson of this thesis, beyond the physical findings themselves, is that representation learning's value is NOT uniform and must be validated per-condition, with cross-validating methods, before being claimed. A single global accuracy number (e.g., \"90% retrieval\" from this project's own earlier, since-corrected results) can look impressive while hiding this exact conditional structure \u2014 or, as happened during this project's development, reflect a methodological artifact rather than a real signal at all."
        ),

        H1("6. Methodological Transparency Notes"),
        Bullet(
          "An earlier evaluation reported inflated numbers (90%+ retrieval, 52 dB PSNR) due to a missing train/test split, a data-loading bug misaligning SAR/optical/label triplets, and a zero-target decoder bug. All three were corrected; every result in this document reflects the corrected pipeline."
        ),
        Bullet(
          "An earlier version of the embedding-level CCA reported a canonical correlation of exactly 1.000 in every group \u2014 this was diagnosed as a classic CCA overfitting artifact (64 embedding dimensions vs. as few as 14\u201335 samples per group) and fixed via PCA dimensionality reduction plus a permutation-test null baseline. This is disclosed as a second concrete example of the same underlying lesson: a superficially strong cross-modal result must be checked against a proper null/control before being trusted."
        ),

        H1("7. Limitations & Threats to Validity"),
        Bullet(
          "Small dataset (172 patches); several per-cluster groups have n < 10, so cluster-level findings are suggestive, strengthened by (but not immune from small-sample noise despite) cross-method agreement."
        ),
        Bullet(
          "GMM clustering was fit on the full dataset (train+test) before any split existed \u2014 a milder residual leakage than the corrected bugs above."
        ),
        Bullet(
          "The Physical Feature Decoder's targets are the same descriptors used to derive GMM labels that shaped the embedding space \u2014 a partial circularity noted throughout."
        ),
        Bullet(
          "The entropy proxy is a simplified dual-pol approximation, not rigorous Cloude\u2013Pottier polarimetric entropy."
        ),
        Bullet(
          "CCA and the baseline comparison were run on a single reference split, not averaged across the 3 multi-seed runs in Section 4.1."
        ),

        H1("8. Conclusion"),
        P(
          "This thesis developed a SupCon-trained SAR-optical representation learning pipeline, validated its stability across multiple random splits, and \u2014 via CCA and a controlled baseline comparison, both cross-checked against downstream task performance \u2014 established that its value over simple hand-crafted descriptors is conditional on land-cover type, not universal. Dense-canopy conditions (Cluster 0) show no cross-modal benefit from representation learning across three independent analyses; sparser-canopy conditions (Cluster 1) show clear benefit across the same three. This conditional, physically-explainable, multiply-validated account is the thesis's central contribution."
        ),

        H1("9. Future Work"),
        Bullet("Extend CCA and baseline comparison across all 3 multi-seed splits."),
        Bullet("Expand the dataset to raise per-cluster sample sizes above the current interpretability floor."),
        Bullet(
          "Investigate why Clusters 5 and 6 show partial disagreement between CCA and task-level performance \u2014 possibly indicating CCA's linear-correlation assumption misses non-linear structure the decoder can still exploit."
        ),
        Bullet(
          "Obtain quad-pol/SLC SAR data to replace the entropy proxy with rigorous polarimetric decomposition."
        )
      ]
    }
  ]
});

// Compile and write document to disk
Packer.toBuffer(doc)
  .then((buffer) => {
    fs.writeFileSync("final_thesis_v2.docx", buffer);
    console.log("Successfully generated final_thesis_v2.docx!");
  })
  .catch((err) => {
    console.error("Error generating Word document:", err);
  });