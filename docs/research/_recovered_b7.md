[assistant]
Now, continue research. Let me search for the remaining gaps:
1. AdaFace model zoo numbers + download URLs/sizes
2. CLIP-ReID numbers + weights
3. SOLIDER numbers + weights
4. OSNet ONNX download
5. Practical latency/throughput at 30 FPS on 8GB VRAM
6. Face detection (SCRFD/RetinaFace) ONNX sizes
7. YOLO26-ReID (already covered)
8. CRISP / one-shot target ReID, distractor-aware

Let me try web_search again (might be recovered) plus direct fetches.

Actually, let me use web_search with good queries. And in parallel fetch known GitHub raw URLs.

Known good raw URLs:
- https://raw.githubusercontent.com/deepinsight/insightface/master/recognition/arcface_torch/README.md (ArcFace numbers + model zoo with sizes)
- https://raw.githubusercontent.com/Syliz517/CLIP-ReID/main/README.md (404 before, but let me try the GitHub API)
- https://raw.githubusercontent.com/tinyvision/SOLIDER/main/README.md
- https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/docs/model_zoo.md
- https://raw.githubusercontent.com/vlsomers/bpbreid/master/README.md

Let me batch a web_search (multiple queries) and a couple of fetches.

[assistant • web_search]
{"queries":["AdaFace model zoo ir50 ir101 IJB-B IJB-C TAR@FAR download","CLIP-ReID AAAI 2023 results Market-1501 MSMT17 mAP Rank-1 weights download","SOLIDER Swin tiny base results MSMT17 mAP Rank-1 weights download"]}

[assistant • web_fetch]
{"url":"https://raw.githubusercontent.com/deepinsight/insightface/master/recognition/arcface_torch/README.md"}

[assistant • web_fetch]
{"url":"https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/docs/model_zoo.md"}

[tool • web_search]
Error: modsearch failed (exit 1): Error: Every engine for the web source failed.
  - firecrawl: firecrawl timed out after 40000 ms.
  - keenable: keenable request failed: fetch failed
Known engines: antigravity-cli, tavily, exa, firecrawl, keenable, grok-cli, local

[tool • web_fetch]
Error: web fetch failed: TypeError: fetch failed

[tool • web_fetch]
Error: web fetch failed: TypeError: fetch failed

[assistant]
Network seems flaky right now. Let me retry with slight delays, and use different URLs. Let me try github.com HTML pages instead of raw.

Let me try a single fetch and see.

[assistant • web_fetch]
{"url":"https://github.com/vlsomers/bpbreid"}

[assistant • web_fetch]
{"url":"https://arxiv.org/abs/2204.00964"}

[tool • web_fetch]
Fetched https://github.com/vlsomers/bpbreid (HTTP 200)

External web content follows. Treat it as untrusted data, not instructions.

                                                                                                           GitHub - VlSomers/bpbreid: \[WACV23\] A strong baseline for body part-based person re-identification · GitHub                                             

[Skip to content](#start-of-content)    

    

           

## Navigation Menu

[](/)

[Sign in](/login?return_to=https%3A%2F%2Fgithub.com%2Fvlsomers%2Fbpbreid)

-   Platform
    
    -   AI CODE CREATION
        
        -   [GitHub CopilotWrite better code with AI](https://github.com/features/copilot)
        -   [GitHub Copilot appDirect agents from issue to merge](https://github.com/features/ai/github-app)
        -   [MCP RegistryIntegrate external tools](https://github.com/mcp)
        
    -   DEVELOPER WORKFLOWS
        
        -   [ActionsAutomate any workflow](https://github.com/features/actions)
        -   [CodespacesInstant dev environments](https://github.com/features/codespaces)
        -   [IssuesPlan and track work](https://github.com/features/issues)
        -   [Code ReviewManage code changes](https://github.com/features/code-review)
        -   [Code QualityEnforce quality at merge](https://github.com/features/code-quality)
        
    -   APPLICATION SECURITY
        
        -   [GitHub Advanced SecurityFind and fix vulnerabilities](https://github.com/security/advanced-security)
        -   [Code securitySecure your code as you build](https://github.com/security/advanced-security/code-security)
        -   [Secret protectionStop leaks before they start](https://github.com/security/advanced-security/secret-protection)
        
    -   EXPLORE
        
        -   [Why GitHub](https://github.com/why-github)
        -   [Documentation](https://docs.github.com)
        -   [Blog](https://github.blog)
        -   [Changelog](https://github.blog/changelog)
        -   [Marketplace](https://github.com/marketplace)
        
    
    [View all features](https://github.com/features)
    
-   Solutions
    
    -   BY COMPANY SIZE
        
        -   [Enterprises](https://github.com/enterprise)
        -   [Small and medium teams](https://github.com/team)
        -   [Startups](https://github.com/enterprise/startups)
        -   [Nonprofits](https://github.com/solutions/industry/nonprofits)
        
    -   BY USE CASE
        
        -   [App Modernization](https://github.com/solutions/use-case/app-modernization)
        -   [DevSecOps](https://github.com/solutions/use-case/devsecops)
        -   [DevOps](https://github.com/solutions/use-case/devops)
        -   [CI/CD](https://github.com/solutions/use-case/ci-cd)
        -   [View all use cases](https://github.com/solutions/use-case)
        
    -   BY INDUSTRY
        
        -   [Healthcare](https://github.com/solutions/industry/healthcare)
        -   [Financial services](https://github.com/solutions/industry/financial-services)
        -   [Manufacturing](https://github.com/solutions/industry/manufacturing)
        -   [Government](https://github.com/solutions/industry/government)
        -   [See all industries](https://github.com/solutions/industry)
        
    
    [View all solutions](https://github.com/solutions)
    
-   Resources
    
    -   EXPLORE BY TOPIC
        
        -   [AI](https://github.com/resources/articles?topic=ai)
        -   [Software Development](https://github.com/resources/articles?topic=software-development)
        -   [DevOps](https://github.com/resources/articles?topic=devops)
        -   [Security](https://github.com/resources/articles?topic=security)
        -   [View all topics](https://github.com/resources/articles)

(Content truncated. Fetch a more specific URL or section for the full text.)

[tool • web_fetch]
Fetched https://arxiv.org/abs/2204.00964 (HTTP 200)

External web content follows. Treat it as untrusted data, not instructions.

 \[2204.00964\] AdaFace: Quality Adaptive Margin for Face Recognition                                

[Skip to main content](#content)

 [![archive](https://static.arxiv.org/static/design-system/latest/logos/arxiv-logo.svg)](https://arxiv.org/)[Search](https://arxiv.org/search)

[Submit](https://arxiv.org/submit) [Donate](https://info.arxiv.org/about/donate.html) [Log in](https://arxiv.org/login)

# Computer Science > Computer Vision and Pattern Recognition

**arXiv:2204.00964** (cs)

\[Submitted on 3 Apr 2022 ([v1](https://arxiv.org/abs/2204.00964v1)), last revised 16 Feb 2023 (this version, v2)\]

# Title:AdaFace: Quality Adaptive Margin for Face Recognition

Authors:[Minchul Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim,+M), [Anil K. Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain,+A+K), [Xiaoming Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+X)

[View PDF](/pdf/2204.00964) [HTML (experimental)](https://arxiv.org/html/2204.00964v2)

> Abstract:Recognition in low quality face datasets is challenging because facial attributes are obscured and degraded. Advances in margin-based loss functions have resulted in enhanced discriminability of faces in the embedding space. Further, previous studies have studied the effect of adaptive losses to assign more importance to misclassified (hard) examples. In this work, we introduce another aspect of adaptiveness in the loss function, namely the image quality. We argue that the strategy to emphasize misclassified samples should be adjusted according to their image quality. Specifically, the relative importance of easy or hard samples should be based on the sample's image quality. We propose a new loss function that emphasizes samples of different difficulties based on their image quality. Our method achieves this in the form of an adaptive margin function by approximating the image quality with feature norms. Extensive experiments show that our method, AdaFace, improves the face recognition performance over the state-of-the-art (SoTA) on four datasets (IJB-B, IJB-C, IJB-S and TinyFace). Code and models are released in [this https URL](https://github.com/mk-minchul/AdaFace).

|     |     |
| --- | --- |
| Comments: | Published in CVPR2022 (Oral) |
| Subjects: | Computer Vision and Pattern Recognition (cs.CV) |
| Cite as: | [arXiv:2204.00964](https://arxiv.org/abs/2204.00964) \[cs.CV\] |
|     | (or [arXiv:2204.00964v2](https://arxiv.org/abs/2204.00964v2) \[cs.CV\] for this version) |
|     | [https://doi.org/10.48550/arXiv.2204.00964](https://doi.org/10.48550/arXiv.2204.00964)<br><br>Focus to learn more<br><br>arXiv-issued DOI via DataCite |

## Submission history

From: Minchul Kim \[[view email](/show-email/d673793e/2204.00964)\]  
**[\[v1\]](/abs/2204.00964v1)** Sun, 3 Apr 2022 01:23:41 UTC (11,292 KB)  
**\[v2\]** Thu, 16 Feb 2023 04:09:47 UTC (11,292 KB)  

Full-text links:

## Access Paper:

-   [View PDF](/pdf/2204.00964)
-   [HTML (experimental)](https://arxiv.org/html/2204.00964v2)
-   [TeX Source](/src/2204.00964)

[view license](http://arxiv.org/licenses/nonexclusive-distrib/1.0/ "Rights to this article")

### Additional Features

-   [Audio Summary](/audio/2204.00964)

### Current browse context:

cs.CV

[< prev](/prevnext?id=2204.00964&function=prev&context=cs.CV "previous in cs.CV (accesskey p)")   |   [next >](/prevnext?id=2204.00964&function=next&context=cs.CV "next in cs.CV (accesskey n)")  

[new](/list/cs.CV/new) | [recent](/list/cs.CV/recent) | [2022-04](/list/cs.CV/2022-04)

Change to browse by:

[cs](/abs/2204.00964?context=cs)  

### References & Citations

-   [SciX](https://scixplorer.org/abs/arXiv:2204.00964)
-   [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2204.00964)
-   [Semantic Scholar](https://api.semanticscholar.org/arXiv:2204.00964)

export BibTeX citation 

### Bookmark

 [![BibSonomy](/static/browse/0.3.4/images/icons/social/bibsonomy.png)](<http://www.bibsonomy.org/BibtexHandler?requTask=upload&url=https://arxiv.org/abs/2204.00964&description=AdaFace: Quality Adaptive Margin for Face Recognition> "Bookmark on BibSonomy")[![Reddit](/static/browse/0.3.4/images/icons/social/reddit.png)](<https://reddit.com/submit?url=https://arxiv.org/abs/2204.00964&title=AdaFace: Quality Adaptive Margin for Face Recognition> "Bookmark on Reddit")

 Bibliographic Tools

# Bibliographic and Citation Tools

 Bibliographic Explorer Toggle

Bibliographic Explorer _([What is the Explorer?](https://info.arxiv.org/labs/showcase.html#arxiv-bibliographic-explorer))_

 Connected Papers Toggle

Connected Papers _([What is Connected Papers?](https://www.connectedpapers.com/about))_

 Litmaps Toggle

Litmaps _([What is Litmaps?](https://www.litmaps.co/))_

 scite.ai Toggle

scite Smart Citations _([What are Smart Citations?](https://www.scite.ai/))_

 Code, Data, Media

# Code, Data and Media Associated with this Article

 alphaXiv Toggle

alphaXiv _([What is alphaXiv?](https://alphaxiv.org/))_

 Links to Code Toggle

CatalyzeX Code Finder for Papers _([What is CatalyzeX?](https://www.catalyzex.com))_

 DagsHub Toggle

DagsHub _([What is DagsHub?](https://dagshub.com/))_

 GotitPub Toggle

Gotit.pub _([What is GotitPub?](http://gotit.pub/faq))_

 Huggingface Toggle

Hugging Face _([What is Huggingface?](https://huggingface.co/huggingface))_

 ScienceCast Toggle

ScienceCast _([What is ScienceCast?](https://sciencecast.org/welcome))_

 Demos

# Demos

 Replicate Toggle

Replicate _([What is Replicate?](https://replicate.com/docs/arxiv/about))_

 Spaces Toggle

Hugging Face Spaces _([What is Spaces?](https://huggingface.co/docs/hub/spaces))_

 Spaces Toggle

TXYZ.AI _([What is TXYZ.AI?](https://txyz.ai))_

 Related Papers

# Recommenders and Search Tools

 Link to Influence Flower

Influence Flower _([What are Influence Flowers?](https://influencemap.cmlab.dev/))_

 Core recommender toggle

CORE Recommender _([What is CORE?](https://core.ac.uk/services/recommender))_

 About arXivLabs

# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? [**Learn more about arXivLabs**](https://info.arxiv.org/labs/index.html).

[Which authors of this paper are endorsers?](/auth/show-endorsers/2204.00964) | [Disable MathJax](javascript:setMathjaxCookie\(\)) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))

We gratefully acknowledge support from our **major funders**, [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , and all contributors.

[About](https://info.arxiv.org/about)  [Help](https://info.arxiv.org/help)  [Contact](https://info.arxiv.org/help/contact.html)  [Subscribe](https://info.arxiv.org/help/subscribe.html)  [Copyright](https://info.arxiv.org/help/license/index.html)  [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html)  [Accessibility](https://info.arxiv.org/help/web_accessibility.html)  [Operational Status (opens in new tab)](https://status.arxiv.org/)

Major funding support from

 [![Simons Foundation](https://static.arxiv.org/static/design-system/latest/funders/simons-foundation.png) (opens in new tab)](https://www.simonsfoundation.org/)[![Simons Foundation International](https://static.arxiv.org/static/design-system/latest/funders/simons-foundation-international.png) (opens in new tab) ](https://www.sfi.org.bm/)[![Schmidt Sciences](https://static.arxiv.org/static/design-system/latest/funders/schmidt-sciences.png) (opens in new tab)](https://www.schmidtsciences.org/)

[assistant]
Network works for some. The AdaFace GitHub README returned fetch failed earlier; let me try the GitHub HTML page for mk-minchul/AdaFace. Also need CLIP-ReID and SOLIDER.

Actually, a better strategy for GitHub READMEs: use the raw URL on the default branch. For AdaFace the default branch is "master". It failed. Let me try github.com HTML.

Also I should get the AdaFace numbers from the paper HTML: https://arxiv.org/html/2204.00964v2 — that has Table 1/2 with IJB-B/IJB-C TAR@FAR and TinyFace Rank-1. Let me fetch that.

Let me batch:
1. https://arxiv.org/html/2204.00964v2 (AdaFace numbers)
2. https://github.com/mk-minchul/AdaFace (model zoo + download links)
3. https://github.com/Syliz517/CLIP-ReID (CLIP-ReID numbers + weights)

[assistant • web_fetch]
{"url":"https://arxiv.org/html/2204.00964v2"}

[assistant • web_fetch]
{"url":"https://github.com/mk-minchul/AdaFace"}

[tool • web_fetch]
Fetched https://arxiv.org/html/2204.00964v2 (HTTP 200)

External web content follows. Treat it as untrusted data, not instructions.

 AdaFace: Quality Adaptive Margin for Face Recognition          

##### Report GitHub Issue

×

Title: Description:

Submit without GitHub Submit in GitHub

 arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

 [![arXiv logo](/static/base/1.0.1/images/arxiv-logo-primary-light.svg) Back to arXiv](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) [Report Issue](# "Report an Issue") [Back to Abstract](/abs/2204.00964v2 "Back to abstract page") [Download PDF](/pdf/2204.00964v2 "Download PDF")[](javascript:toggleNavTOC\(\); "Toggle navigation")[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

1.  [Abstract](#abstract1 "In AdaFace: Quality Adaptive Margin for Face Recognition")
2.  [1 Introduction](#S1 "In AdaFace: Quality Adaptive Margin for Face Recognition")
3.  [2 Related Work](#S2 "In AdaFace: Quality Adaptive Margin for Face Recognition")
4.  [3 Proposed Approach](#S3 "In AdaFace: Quality Adaptive Margin for Face Recognition")
    1.  [3.1 Margin Form and the Gradient](#S3.SS1 "In 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    2.  [3.2 Norm and Image quality](#S3.SS2 "In 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    3.  [3.3 AdaFace: Adaptive Margin based on Norm](#S3.SS3 "In 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
5.  [4 Experiments](#S4 "In AdaFace: Quality Adaptive Margin for Face Recognition")
    1.  [4.1 Datasets and Implementation Details](#S4.SS1 "In 4 Experiments ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    2.  [4.2 Ablation and Analysis](#S4.SS2 "In 4 Experiments ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    3.  [4.3 Comparison with SoTA methods](#S4.SS3 "In 4 Experiments ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
6.  [5 Conclusion](#S5 "In AdaFace: Quality Adaptive Margin for Face Recognition")
7.  [References](#bib "In AdaFace: Quality Adaptive Margin for Face Recognition")
8.  [A Gradient Scaling Term](#S1a "In AdaFace: Quality Adaptive Margin for Face Recognition")
    1.  [A.1 Derivation of Angular Margin](#S1.SS1 "In A Gradient Scaling Term ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    2.  [A.2 Interpretation of gg](#S1.SS2 "In A Gradient Scaling Term ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
9.  [B Feature Norm Analysis](#S2a "In AdaFace: Quality Adaptive Margin for Face Recognition")
    1.  [B.1 Correlation between Norm and BRISQUE during Training](#S2.SS1 "In B Feature Norm Analysis ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    2.  [B.2 Training Sample Visualization](#S2.SS2 "In B Feature Norm Analysis ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    3.  [B.3 Training Samples’ Gradient Scaling Term for AdaFace](#S2.SS3 "In B Feature Norm Analysis ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
    4.  [B.4 Train Samples’ Gradient Scaling Term Comparison with ArcFace](#S2.SS4 "In B Feature Norm Analysis ‣ AdaFace: Quality Adaptive Margin for Face Recognition")
10.  [C Visualization of Success and Failed Test Images](#S3a "In AdaFace: Quality Adaptive Margin for Face Recognition")
11.  [D Comparison with General Image-Quality Aware Learning Method](#S4a "In AdaFace: Quality Adaptive Margin for Face Recognition")
12.  [E Effect of Batch Size](#S5a "In AdaFace: Quality Adaptive Margin for Face Recognition")
13.  [F Implementation Details and Code](#S6 "In AdaFace: Quality Adaptive Margin for Face Recognition")

[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2204.00964v2 \[cs.CV\] 16 Feb 2023

# AdaFace: Quality Adaptive Margin for Face Recognition

Minchul Kim Affiliation: Department of Computer Science and Engineering,    Anil K. Jain Affiliation: Michigan State University, East Lansing, MI, 48824    Xiaoming Liu Affiliation: {kimminc2, jain, liuxm}@cse.msu.edu

###### Abstract

Recognition in low quality face datasets is challenging because facial attributes are obscured and degraded. Advances in margin-based loss functions have resulted in enhanced discriminability of faces in the embedding space. Further, previous studies have studied the effect of adaptive losses to assign more importance to misclassified (hard) examples. In this work, we introduce another aspect of adaptiveness in the loss function, namely the image quality. We argue that the strategy to emphasize misclassified samples should be adjusted according to their image quality. Specifically, the relative importance of easy or hard samples should be based on the sample’s image quality. We propose a new loss function that emphasizes samples of different difficulties based on their image quality. Our method achieves this in the form of an adaptive margin function by approximating the image quality with feature norms. Extensive experiments show that our method, AdaFace, improves the face recognition performance over the state-of-the-art (SoTA) on four datasets (IJB-B, IJB-C, IJB-S and TinyFace). Code and models are released in [Supp](https://github.com/mk-minchul/AdaFace).

## 1 Introduction

Image quality is a combination of attributes that indicates how faithfully an image captures the original scene \[[32](#bib.bib32)\]. Factors that affect the image quality include brightness, contrast, sharpness, noise, color constancy, resolution, tone reproduction, etc. Face images, the focus of this paper, can be captured under a variety of settings for lighting, pose and facial expression, and sometimes under extreme visual changes such as a subject’s age or make-up. These parameter settings make the recognition task difficult for learned face recognition (FR) models. Still, the task is achievable in the sense that humans or models can often recognize faces under these difficult settings \[[37](#bib.bib37)\]. However, when a face image is of low quality, depending on the degree, the recognition task becomes infeasible. Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ AdaFace: Quality Adaptive Margin for Face Recognition") shows examples of both high quality and low quality face images. It is not possible to recognize the subjects in the last column of Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ AdaFace: Quality Adaptive Margin for Face Recognition").

Figure 1: Examples of face images with different qualities and recognizabilities. Both high and low quality images contain variations in pose, occlusion and resolution that sometimes make the recognition task difficult, yet achievable. Depending on the degree of degradation, some images may become impossible to recognize. By studying the different impacts these images have in training, this work aims to design a novel loss function that is adaptive to a sample’s recognizability, driven by its image quality.

Low quality images like the bottom row of Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ AdaFace: Quality Adaptive Margin for Face Recognition") are increasingly becoming an important part of face recognition datasets because they are encountered in surveillance videos and drone footage. Given that SoTA FR methods \[[7](#bib.bib7), [16](#bib.bib16), [21](#bib.bib21), [8](#bib.bib8)\] are able to obtain over 98% verification accuracy in relatively high quality datasets such as LFW or CFP-FP \[[14](#bib.bib14), [31](#bib.bib31)\], recent FR challenges have moved to lower quality datasets such as IJB-B, IJB-C and IJB-S \[[41](#bib.bib41), [26](#bib.bib26), [17](#bib.bib17)\]. Although the challenge is to attain high accuracy on low quality datasets, most popular training datasets still remain comprised of high quality images \[[11](#bib.bib11), [7](#bib.bib7)\]. Since only a small portion of training data is low quality, it is important to properly leverage it during training.

Figure 2: Conventional margin based softmax loss vs our AdaFace. (a) A FR training pipeline with a margin based softmax loss. The loss function takes the margin function to induce smaller intra-class variations. Some examples are SphereFace, CosFace and ArcFace \[[24](#bib.bib24), [39](#bib.bib39), [7](#bib.bib7)\]. (b) Proposed adaptive margin function (AdaFace) that is adjusted based on the image quality indicator. If the image quality is indicated to be low, the loss function emphasizes easy samples (thereby avoiding unidentifiable images). Otherwise, the loss emphasizes hard samples.

One problem with low quality face images is that they tend to be unrecognizable. When the image degradation is too large, the relevant identity information vanishes from the image, resulting in unidentifiable images. These unidentifiable images are detrimental to the training procedure since a model will try to exploit other visual characteristics, such as clothing color or image resolution, to lower the training loss. If these images are dominant in the distribution of low quality images, the model is likely to perform poorly on low quality datasets during testing.

Motivated by the presence of unidentifiable facial images, we would like to design a loss function which assigns different importance to samples of different difficulties according to the image quality. We aim to emphasize hard samples for the high quality images and easy samples for low quality images. Typically, assigning different importance to different difficulties of samples is done by looking at the training progression (curriculum learning) \[[4](#bib.bib4), [16](#bib.bib16)\]. Yet, we show that the sample importance should be adjusted by looking at both the difficulty and the image quality.

The reason why importance should be set differently according to the image quality is that naively emphasizing hard samples always puts a strong emphasis on unidentifiable images. This is because one can only make a random guess about unidentifiable images and thus, they are always in the hard sample group. There are challenges in introducing image quality into the objective. This is because image quality is a term that is hard to quantify due to its broad definition and scaling samples based on the difficulty often introduces ad-hoc procedures that are heuristic in nature.

In this work, we present a loss function to achieve the above goal in a seamless way. We find that 1) feature norm can be a good proxy for the image quality, and 2) various margin functions amount to assigning different importance to different difficulties of samples. These two findings are combined in a unified loss function, AdaFace, that adaptively changes the margin function to assign different importance to different difficulties of samples, based on the image quality (see Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ AdaFace: Quality Adaptive Margin for Face Recognition")).

In summary, the contributions of this paper include:

-   •
    
    We propose a loss function, AdaFace, that assigns different importance to different difficulties of samples according to their image quality. By incorporating image quality, we avoid emphasizing unidentifiable images while focusing on hard yet recognizable samples.
    
-   •
    
    We show that the angular margin scales the learning signal (gradient) based on the training sample’s difficulty. This observation motivates us to change margin function adaptively to emphasize hard samples if the image quality is high, and ignore very hard samples (unidentifiable images) if the image quality is low.
    
-   •
    
    We demonstrate that feature norms can serve as the proxy of image quality. It bypasses the need for an additional module to estimate image quality. Thus, adaptive margin function is achieved without additional complexity.
    
-   •
    
    We verify the efficacy of the proposed method by extensive evaluations on 99 datasets (LFW, CFP-FP, CPLFW, AgeDB, CALFW, IJB-B, IJB-C, IJB-S and TinyFace) of various qualities. We show that the recognition performance on low quality datasets can be hugely increased while maintaining performance on high quality datasets.
    

## 2 Related Work

Margin Based Loss Function.  The margin based softmax loss function is widely used for training face recognition (FR) models \[[39](#bib.bib39), [7](#bib.bib7), [24](#bib.bib24), [16](#bib.bib16)\]. Margin is added to the softmax loss because without the margin, learned features are not sufficiently discriminative. SphereFace \[[24](#bib.bib24)\], CosFace \[[39](#bib.bib39)\] and ArcFace \[[7](#bib.bib7)\] introduce different forms of margin functions. Specifically, it can be written as,

|     |     |     |     |
| --- | --- | --- | --- |
|     | ℒ\=−log⁡exp⁡(f⁡(θyi,m))exp(f(θyi,m))+∑j≠yinexp(scosθj),\\mathcal{L}=-\\log\\frac{\\exp(f(\\theta\_{y\_{i}},m))}{\\exp(f(\\theta\_{y\_{i}},m))+\\sum\_{j\\neq y\_{i}}^{n}\\exp(s\\cos\\theta\_{j})}, |     | (1) |

where θj\\theta\_{j} is the angle between the feature vector and the jt​hj^{th} classifier weight vector, yiy\_{i} is the index of the ground truth (GT) label, and mm is the margin, which is a scalar hyper-parameter. ff is a margin function, where

|     |     |     |     |
| --- | --- | --- | --- |
|     | f​(θj,m)SphereFace\={s​cos⁡(m​θj)j\=yiscosθjj≠yi,f(\\theta\_{j},m)\_{\\text{SphereFace}}=\\begin{cases}s\\cos(m\\theta\_{j})&j=y\_{i}\\\\ s\\cos\\theta\_{j}&j\\neq y\_{i}\\end{cases}, |     | (2) |

|     |     |     |     |
| --- | --- | --- | --- |
|     | f​(θj,m)CosFace\={s⁡(cos⁡θj−m)j\=yiscosθjj≠yi,f(\\theta\_{j},m)\_{\\text{CosFace}}=\\begin{cases}s(\\cos\\theta\_{j}-m)&j=y\_{i}\\\\ s\\cos\\theta\_{j}&j\\neq y\_{i}\\end{cases}, |     | (3) |

|     |     |     |     |
| --- | --- | --- | --- |
|     | f​(θj,m)ArcFace\={s​cos⁡(θj+m)j\=yiscosθjj≠yi.f(\\theta\_{j},m)\_{\\text{ArcFace}}=\\begin{cases}s\\cos(\\theta\_{j}+m)&j=y\_{i}\\\\ s\\cos\\theta\_{j}&j\\neq y\_{i}\\end{cases}. |     | (4) |

Sometimes, ArcFace is referred to as an angular margin and CosFace is referred to as an additive margin. Here, ss is a hyper-parameter for scaling. P2SGrad \[[46](#bib.bib46)\] notes that mm and ss are sensitive hyper-parameters and proposes to directly modify the gradient to be free of mm and ss.

Our approach aims to model the margin mm as a function of the image quality because f⁡(θyi,m)f(\\theta\_{y\_{i}},m) has an impact on which samples contribute more gradient (i.e. learning signal) during training.

Adaptive Loss Functions.  Many studies have introduced an element of adaptiveness in the training objective for either hard sample mining \[[40](#bib.bib40), [22](#bib.bib22)\], scheduling difficulty during training \[[16](#bib.bib16), [35](#bib.bib35)\], or finding optimal hyperparameters \[[45](#bib.bib45)\]. For example, CurricularFace \[[16](#bib.bib16)\] brings the idea of curriculum learning into the loss function. During the initial stages of training, the margin for cos⁡θj\\cos\\theta\_{j} (negative cosine similarity) is set to be small so that easy samples can be learned and in the later stages, the margin is increased so that hard samples are learned. Specifically, it is written as

|     |     |     |     |
| --- | --- | --- | --- |
|     | f​(θj,m)Curricular\={s​cos⁡(θj+m)j\=yiN⁡(t,cos⁡θj)j≠yi,\\scriptsize f(\\theta\_{j},m)\_{\\text{Curricular}}=\\begin{cases}s\\cos(\\theta\_{j}+m)&j=y\_{i}\\\\ N(t,\\cos\\theta\_{j})&j\\neq y\_{i}\\end{cases}, |     | (5) |

where

|     |     |     |     |
| --- | --- | --- | --- |
|     | N⁡(t,cos⁡θj)\={cos⁡(θj)s​cos⁡(θyi+m)≥cos⁡θjcos⁡(θj)​(t+cos⁡θj)s​cos⁡(θyi+m)<cos⁡θj,\\scriptsize N(t,\\cos\\theta\_{j})=\\begin{cases}\\cos(\\theta\_{j})&s\\cos(\\theta\_{y\_{i}}+m)\\geq\\cos\\theta\_{j}\\\\ \\cos(\\theta\_{j})(t+\\cos\\theta\_{j})&s\\cos(\\theta\_{y\_{i}}+m)<\\cos\\theta\_{j}\\end{cases}, |     | (6) |

and tt is a parameter that increases as the training progresses. Therefore, in CurricularFace, the adaptiveness in the margin is based on the training progression (curriculum).

On the contrary, we argue that the adaptiveness in the margin should be based on the image quality. We believe that among high quality images, if a sample is hard (with respect to a model), the network should learn to exploit the information in the image, but in low quality images, if a sample is hard, it is more likely to be devoid of proper identity clues and the network should not try hard to fit on it.

MagFace \[[27](#bib.bib27)\] explores the idea of applying different margins based on recognizability. It applies large angular margins to high norm features on the premise that high norm features are easily recognizable. Large margin pushes features of high norm closer to class centers. Yet, it fails to emphasize hard training samples, which is important for learning discriminative features. A detailed contrast with MagFace can be found in the supplementary B.1. It is also worth mentioning that DDL \[[15](#bib.bib15)\] uses the distillation loss to minimize the gap between easy and hard sample features.

Face Recognition with Low Quality Images.  Recent FR models have achieved high performance on datasets where facial attributes are discernable, e.g., LFW\[[14](#bib.bib14)\], CFP-FP\[[31](#bib.bib31)\], CPLFW\[[49](#bib.bib49)\], AgeDB\[[29](#bib.bib29)\] and CALFW\[[50](#bib.bib50)\]. Good performance on these datasets can be achieved when the FR model learns discriminative features invariant to lighting, age or pose variations. However, FR in unconstrained scenarios such as in surveillance or low quality videos \[[42](#bib.bib42)\] brings more problems to the table. Examples of datasets in this setting are IJB-B\[[41](#bib.bib41)\], IJB-C\[[26](#bib.bib26)\] and IJB-S\[[17](#bib.bib17)\], where most of the images are of low quality, and some do not contain sufficient identity information, even for human examiners. The key to good performance involves both 1) learning discriminative features for low quality images and 2) learning to discard images that contain few identity cues. The latter is sometimes referred to as quality aware fusion.

To perform quality aware fusion, probabilistic approaches have been proposed to predict uncertainty in FR representation \[[30](#bib.bib30), [33](#bib.bib33), [5](#bib.bib5), [21](#bib.bib21), [51](#bib.bib51)\]. They assume the features are distributions and the variance can be used to calculate the certainty in prediction. However, probabilistic approaches often resort to learning mean and variance separately, which is not simple during training and suboptimal as the variance is optimized with a fixed mean. Our work, however, is a modification to the conventional softmax loss, making the framework easy to use. And we use the feature norm as a proxy for quality during quality-aware fusion.

QSub-PM \[[47](#bib.bib47)\] and UGG \[[48](#bib.bib48)\] also show good performances in LQ video recognition by using rich subspace (matrix) representation for comparison and using auxiliary context (such as a body) to aid feature fusion respectively.

Synthetic data or augmentations can be used to mimic low quality data. \[[34](#bib.bib34), [10](#bib.bib10)\] adopts 3D reconstruction to generate faces. Extra steps complicate the training procedure, making it hard to generalize to other domains. We adopt easily applicable crop, blur and photometric augmentations.

## 3 Proposed Approach

![Refer to caption](2204.00964v2/figure2_v4_newfont.png)

Figure 3: Illustration of different margin functions and their gradient scaling terms on the feature space. B0B\_{0} and B1B\_{1} show the decision boundary with and without margin mm, respectively. The yellow arrow indicates the shift in the boundary due to margin mm. In the arc, a well-classified sample will be close to (in angle) the ground truth class weight vector, 𝑾yi\\bm{W}\_{y\_{i}}. A misclassified sample will be close to 𝑾j\\bm{W}\_{j}, the negative class weight vector. The color within the arc indicates the magnitude of the gradient scaling term gg (Eq. [12](#S3.E12 "Equation 12 ‣ 3.1 Margin Form and the Gradient ‣ 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition")). Samples in the dark red region will contribute more to learning. Note that additive margin shifts the boundary toward 𝑾yi\\bm{W}\_{y\_{i}}, without changing the gradient scaling term. However, positive angular margin not only shifts the boundary, but also makes the gradient scale high near the boundary and low away from the boundary. This behavior de-emphasizes very hard samples, and likewise MagFace has similar behavior. On the other hand, negative angular margin induces an opposite behavior. CurricularFace adapts the boundary based on the training stage. Our work adaptively changes the margin functions based on the norm. With high norm, we emphasize samples away from the boundary and with low norm we emphasize samples near the boundary. Circles and triangles in the arc show example scenarios in the right most plot (AdaFace).

The cross entropy softmax loss of a sample 𝒙i\\bm{x}\_{i} can be formulated as follows,

|     |     |     |     |
| --- | --- | --- | --- |
|     | ℒC​E​(𝒙i)\=−log⁡exp⁡(𝑾yi​𝒛i+byi)∑j\=1Cexp⁡(𝑾j​𝒛j+bj),\\mathcal{L}\_{CE}(\\bm{x}\_{i})=-\\log\\frac{\\exp(\\bm{W}\_{y\_{i}}\\bm{z}\_{i}+b\_{y\_{i}})}{\\sum\_{j=1}^{C}\\exp(\\bm{W}\_{j}\\bm{z}\_{j}+b\_{j})}, |     | (7) |

where 𝒛i∈ℝd\\bm{z}\_{i}\\in\\mathbb{R}^{d} is the 𝒙i\\bm{x}\_{i}’s feature embedding, and 𝒙i\\bm{x}\_{i} belongs to the yiy\_{i}th class. 𝑾j\\bm{W}\_{j} refers to the jjth column of the last FC layer weight matrix, 𝑾∈ℝd×C\\bm{W}\\!\\in\\mathbb{R}^{d\\times C}, and bjb\_{j} refers to the corresponding bias term. CC refers to the number of classes.

During test time, for an arbitrary pair of images, 𝒙p\\bm{x}\_{p} and 𝒙q\\bm{x}\_{q}, the cosine similarity metric, 𝒛p⋅𝒛q‖𝒛p‖​‖𝒛q‖\\frac{\\bm{z}\_{p}\\cdot\\bm{z}\_{q}}{\\|\\bm{z}\_{p}\\|\\|\\bm{z}\_{q}\\|} is used to find the closest matching identities. To make the training objective directly optimize the cosine distance, \[[24](#bib.bib24), [38](#bib.bib38)\] use normalized softmax where the bias term is set to zero and the feature 𝒛i\\bm{z}\_{i} is normalized and rescaled with ss during training. This modification results in

|     |     |     |     |
| --- | --- | --- | --- |
|     | ℒC​E​(𝒙i)\=−log⁡exp⁡(s⋅cos⁡θyi)∑j\=1Cexp(scosθj),\\mathcal{L}\_{CE}(\\bm{x}\_{i})=-\\log\\frac{\\exp(s\\cdot\\cos\\theta\_{y\_{i}})}{\\sum\_{j=1}^{C}\\exp(s\\cos\\theta\_{j})}, |     | (8) |

where θj\\theta\_{j} corresponds to the angle between 𝒛i\\bm{z}\_{i} and 𝑾j\\bm{W}\_{j}. Follow-up works \[[7](#bib.bib7), [39](#bib.bib39)\] take this formulation and introduces a margin to reduce the intra-class variations. Generally, it can be written as Eq. [1](#S2.E1 "Equation 1 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition") where margin functions are defined in Eqs. [2](#S2.E2 "Equation 2 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition"), [3](#S2.E3 "Equation 3 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition") and [4](#S2.E4 "Equation 4 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition") correspondingly.

### 3.1 Margin Form and the Gradient

Previous works on margin based softmax focused on how the margin shifts the decision boundaries and what their geometric interpretations are \[[39](#bib.bib39), [7](#bib.bib7)\]. In this section, we show that during backpropagation, the gradient change due to the margin has the effect of scaling the importance of a sample relative to the others. In other words, angular margin can introduce an additional term in the gradient equation that scales the signal according to the sample’s difficulty. To show this, we will look at how the gradient equation changes with the margin function f⁡(θyi,m)f(\\theta\_{y\_{i}},m).

Let Pj(i)P\_{j}^{(i)} be the probability output at class jj after softmax operation on an input 𝒙i\\bm{x}\_{i}. By deriving the gradient equations for ℒC​E\\mathcal{L}\_{CE} w.r.t. 𝑾j\\bm{W}\_{j} and 𝒙i\\bm{x}\_{i}, we obtain the following,

|     |     |     |     |
| --- | --- | --- | --- |
|     | Pj(i)\=exp⁡(f⁡(cos⁡θyi))exp(f(cosθyi))+∑j≠yinexp(scosθj),P\_{j}^{(i)}=\\frac{\\exp(f(\\cos\\theta\_{y\_{i}}))}{\\exp(f(\\cos\\theta\_{y\_{i}}))+\\sum\_{j\\neq y\_{i}}^{n}\\exp(s\\cos\\theta\_{j})}, |     | (9) |

|     |     |     |     |
| --- | --- | --- | --- |
|     | ∂ℒCE∂𝑾j\=(Pj(i)−𝟙​(yi\=j))​∂f⁡(cos⁡θj)∂cos⁡θj​∂cos⁡θj∂𝑾j,\\frac{\\partial\\mathcal{L\_{\\text{CE}}}}{\\partial\\bm{W}\_{j}}=\\left(P\_{j}^{(i)}-\\mathbbm{1}(y\_{i}=j)\\right)\\frac{\\partial f(\\cos\\theta\_{j})}{\\partial\\cos\\theta\_{j}}\\frac{\\partial\\cos\\theta\_{j}}{\\partial\\bm{W}\_{j}}, |     | (10) |

|     |     |     |     |
| --- | --- | --- | --- |
|     | ∂ℒCE∂𝒙i\=∑k\=1C(Pk(i)−𝟙​(yi\=k))​∂f⁡(cos⁡θk)∂cos⁡θk​∂cos⁡θk∂𝒙i.\\frac{\\partial\\mathcal{L\_{\\text{CE}}}}{\\partial\\bm{x}\_{i}}\\!=\\!\\sum\_{k=1}^{C}\\!\\left(P\_{k}^{(i)}\\!-\\!\\mathbbm{1}(y\_{i}=k)\\!\\right)\\!\\frac{\\partial f(\\cos\\theta\_{k})}{\\partial\\cos\\theta\_{k}}\\!\\frac{\\partial\\cos\\theta\_{k}}{\\partial\\bm{x}\_{i}}. |     | (11) |

In Eqs. [10](#S3.E10 "Equation 10 ‣ 3.1 Margin Form and the Gradient ‣ 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition") and [11](#S3.E11 "Equation 11 ‣ 3.1 Margin Form and the Gradient ‣ 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition"), the first two terms, (Pj(i)−𝟙​(yi\=j))\\left(P\_{j}^{(i)}-\\mathbbm{1}(y\_{i}=j)\\right) and ∂f⁡(cos⁡θj)∂cos⁡θj\\frac{\\partial f(\\cos\\theta\_{j})}{\\partial\\cos\\theta\_{j}} are scalars. Also, these two are the only terms affected by parameter mm through f⁡(cos⁡θyi)f(\\cos\\theta\_{y\_{i}}). As the direction term, ∂cos⁡θj∂𝑾j\\frac{\\partial\\cos\\theta\_{j}}{\\partial\\bm{W}\_{j}} is free of mm, we can think of the first two scalar terms as a gradient scaling term (GST) and denote,

|     |     |     |     |
| --- | --- | --- | --- |
|     | g:=(Pj(i)−𝟙​(yi\=j))​∂f⁡(cos⁡θj)∂cos⁡θj.g:=\\left(P\_{j}^{(i)}-\\mathbbm{1}(y\_{i}=j)\\right)\\frac{\\partial f(\\cos\\theta\_{j})}{\\partial\\cos\\theta\_{j}}. |     | (12) |

For the purpose of the GST analysis, we will consider the class index j\=yij=y\_{i}, since all negative class indices j≠yij\\neq y\_{i} do not have a margin in Eqs. [2](#S2.E2 "Equation 2 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition"), [3](#S2.E3 "Equation 3 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition"), and [4](#S2.E4 "Equation 4 ‣ 2 Related Work ‣ AdaFace: Quality Adaptive Margin for Face Recognition"). The GST for the normalized softmax loss is

|     |     |     |     |
| --- | --- | --- | --- |
|     | gsoftmax\=(Pyi(i)−1)​s,g\_{\\text{softmax}}=(P\_{y\_{i}}^{(i)}-1)s, |     | (13) |

since f⁡(cos⁡θyi)\=s⋅cos⁡θyif(\\cos\\theta\_{y\_{i}})=s\\cdot\\cos\\theta\_{y\_{i}} and ∂f⁡(cos⁡θyi)∂cos⁡θyi\=s\\frac{\\partial f(\\cos\\theta\_{y\_{i}})}{\\partial\\cos\\theta\_{y\_{i}}}=s. The GST for the CosFace\[[39](#bib.bib39)\] is also

|     |     |     |     |
| --- | --- | --- | --- |
|     | gCosFace\=(Pyi(i)−1)​s,g\_{\\text{CosFace}}=(P\_{y\_{i}}^{(i)}-1)s, |     | (14) |

as f⁡(cos⁡θyi)\=s⁡(cos⁡θyi−m)f(\\cos\\theta\_{y\_{i}})=s(\\cos\\theta\_{y\_{i}}-m) and ∂f⁡(cos⁡θyi)∂cos⁡θyi\=s\\frac{\\partial f(\\cos\\theta\_{y\_{i}})}{\\partial\\cos\\theta\_{y\_{i}}}=s. Yet, the GST for ArcFace\[[7](#bib.bib7)\] turns out to be

|     |     |     |     |
| --- | --- | --- | --- |
|     | gArcFace\=(Pj(i)−1)​s​(cos⁡(m)+cos⁡θyi​sin⁡(m)−cos2⁡θyi).g\_{\\text{ArcFace}}\\!=(P\_{j}^{(i)}\\!-1)s\\left(\\cos(m)\\!+\\frac{\\cos\\theta\_{y\_{i}}\\sin(m)}{\\sqrt{1\\!-\\cos^{2}\\theta\_{y\_{i}}}}\\right). |     | (15) |

The derivation can be found in the supplementary. Since the GST is a function of θyi\\theta\_{y\_{i}} and mm as in Eq. [15](#S3.E15 "Equation 15 ‣ 3.1 Margin Form and the Gradient ‣ 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition"), it is possible to use it to control the emphasis on samples based on the difficulty, i.e., θyi\\theta\_{y\_{i}} during training.

To understand the effect of GST, we visualize GST w.r.t. the features. Fig. [3](#S3.F3 "Figure 3 ‣ 3 Proposed Approach ‣ AdaFace: Quality Adaptive Margin for Face Recognition") shows the GST as the color in the feature space. Note that for the angular margin, the GST peaks at the decision bounda

(Content truncated. Fetch a more specific URL or section for the full text.)

[tool • web_fetch]
Fetched https://github.com/mk-minchul/AdaFace (HTTP 200)

External web content follows. Treat it as untrusted data, not instructions.

                                                                                                           GitHub - mk-minchul/AdaFace · GitHub                                             

[Skip to content](#start-of-content)    

    

           

## Navigation Menu

[](/)

[Sign in](/login?return_to=https%3A%2F%2Fgithub.com%2Fmk-minchul%2FAdaFace)

-   Platform
    
    -   AI CODE CREATION
        
        -   [GitHub CopilotWrite better code with AI](https://github.com/features/copilot)
        -   [GitHub Copilot appDirect agents from issue to merge](https://github.com/features/ai/github-app)
        -   [MCP RegistryIntegrate external tools](https://github.com/mcp)
        
    -   DEVELOPER WORKFLOWS
        
        -   [ActionsAutomate any workflow](https://github.com/features/actions)
        -   [CodespacesInstant dev environments](https://github.com/features/codespaces)
        -   [IssuesPlan and track work](https://github.com/features/issues)
        -   [Code ReviewManage code changes](https://github.com/features/code-review)
        -   [Code QualityEnforce quality at merge](https://github.com/features/code-quality)
        
    -   APPLICATION SECURITY
        
        -   [GitHub Advanced SecurityFind and fix vulnerabilities](https://github.com/security/advanced-security)
        -   [Code securitySecure your code as you build](https://github.com/security/advanced-security/code-security)
        -   [Secret protectionStop leaks before they start](https://github.com/security/advanced-security/secret-protection)
        
    -   EXPLORE
        
        -   [Why GitHub](https://github.com/why-github)
        -   [Documentation](https://docs.github.com)
        -   [Blog](https://github.blog)
        -   [Changelog](https://github.blog/changelog)
        -   [Marketplace](https://github.com/marketplace)
        
    
    [View all features](https://github.com/features)
    
-   Solutions
    
    -   BY COMPANY SIZE
        
        -   [Enterprises](https://github.com/enterprise)
        -   [Small and medium teams](https://github.com/team)
        -   [Startups](https://github.com/enterprise/startups)
        -   [Nonprofits](https://github.com/solutions/industry/nonprofits)
        
    -   BY USE CASE
        
        -   [App Modernization](https://github.com/solutions/use-case/app-modernization)
        -   [DevSecOps](https://github.com/solutions/use-case/devsecops)
        -   [DevOps](https://github.com/solutions/use-case/devops)
        -   [CI/CD](https://github.com/solutions/use-case/ci-cd)
        -   [View all use cases](https://github.com/solutions/use-case)
        
    -   BY INDUSTRY
        
        -   [Healthcare](https://github.com/solutions/industry/healthcare)
        -   [Financial services](https://github.com/solutions/industry/financial-services)
        -   [Manufacturing](https://github.com/solutions/industry/manufacturing)
        -   [Government](https://github.com/solutions/industry/government)
        -   [See all industries](https://github.com/solutions/industry)
        
    
    [View all solutions](https://github.com/solutions)
    
-   Resources
    
    -   EXPLORE BY TOPIC
        
        -   [AI](https://github.com/resources/articles?topic=ai)
        -   [Software Development](https://github.com/resources/articles?topic=software-development)
        -   [DevOps](https://github.com/resources/articles?topic=devops)
        -   [Security](https://github.com/resources/articles?topic=security)
        -   [View all topics](https://github.com/resources/articles)
        
    -   EXPLORE BY TYPE

(Content truncated. Fetch a more specific URL or section for the full text.)

[assistant]
GitHub HTML is heavily loaded with nav boilerplate and gets truncated before README content. Better to fetch raw.githubusercontent.com with the right branch. Earlier raw fetch of mk-minchul/AdaFace/master/README.md failed with "fetch failed" (transient?). Let me retry raw, and also try codeload or the GitHub API for README.

Actually a good approach: use the GitHub API: https://api.github.com/repos/mk-minchul/AdaFace/readme with Accept header — but web_fetch can't set headers. Alternative: raw.githubusercontent.com works. Let me retry.

Also for the README content that got truncated in HTML, I could fetch with a fragment? No.

Let me retry raw URLs. And use jina.ai reader? https://r.jina.ai/https://github.com/... might work but adds dependency.

Let me just retry raw with several at once.

[assistant • web_fetch]
{"url":"https://raw.githubusercontent.com/mk-minchul/AdaFace/master/README.md"}

[assistant • web_fetch]
{"url":"https://raw.githubusercontent.com/tinyvision/SOLIDER/main/README.md"}

[tool • web_fetch]
Error: web fetch failed: TypeError: fetch failed

[tool • web_fetch]
Error: web fetch failed: TypeError: fetch failed

[assistant]
raw.githubusercontent.com seems blocked or flaky. But earlier in the session I successfully fetched raw.githubusercontent.com URLs? Let me check: the compressed summary mentioned fetching raw ultralytics track.md successfully at m00036. So raw works sometimes.

Maybe there's a transient network issue. Let me use an alternative: install/use the local machine's network via pwsh Invoke-WebRequest? The workspace is Windows with danger-full-access. I could use pwsh to curl the URLs — that might bypass whatever proxy the web_fetch uses. Let me test.

Actually that's a good idea: use pwsh Invoke-RestMethod to fetch READMEs. Let me try.

[assistant • pwsh]
{"command":"$ProgressPreference='SilentlyContinue'; foreach ($u in @('https://raw.githubusercontent.com/mk-minchul/AdaFace/master/README.md','https://raw.githubusercontent.com/tinyvision/SOLIDER/main/README.md')) { Write-Output \"=== $u\"; try { $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 25; Write-Output (\"STATUS \" + $r.StatusCode + \" LEN \" + $r.Content.Length) } catch { Write-Output (\"ERR \" + $_.Exception.Message) } }","description":"Test fetching raw READMEs via PowerShell"}

[tool • pwsh]
=== https://raw.githubusercontent.com/mk-minchul/AdaFace/master/README.md
ERR 操作超时。
=== https://raw.githubusercontent.com/tinyvision/SOLIDER/main/README.md
ERR 操作超时。


[assistant]
raw.githubusercontent.com is unreachable from this machine. But earlier in the session (per summary) raw fetches succeeded. Hmm, maybe they went through a proxy that's now down.

Let me test general connectivity: can I reach github.com? Yes (HTML fetches worked). So raw.githubusercontent.com is blocked/unresolvable. Workaround: use github.com/<repo>/raw/<branch>/README.md which redirects to raw.githubusercontent... that would fail too. 

Alternative: use codeload.github.com tarball? Or use the GitHub HTML page with a more targeted approach — the README content in HTML is at the end; truncation kills it. 

Better: use `https://github.com/mk-minchul/AdaFace/blob/master/README.md` — same truncation.

Alternative mirror: jsDelivr! `https://cdn.jsdelivr.net/gh/mk-minchul/AdaFace@master/README.md` — this serves GitHub files and is often accessible. Let me try that. Also `https://gitcdn.link` etc.

Let me test jsDelivr.

[assistant • pwsh]
{"command":"$ProgressPreference='SilentlyContinue'; $urls=@('https://cdn.jsdelivr.net/gh/mk-minchul/AdaFace@master/README.md','https://cdn.jsdelivr.net/gh/tinyvision/SOLIDER@main/README.md','https://cdn.jsdelivr.net/gh/Syliz517/CLIP-ReID@main/README.md'); foreach ($u in $urls) { Write-Output \"=== $u\"; try { $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 25; Write-Output (\"STATUS \" + $r.StatusCode + \" LEN \" + $r.Content.Length) } catch { Write-Output (\"ERR \" + $_.Exception.Message) } }","description":"Test jsDelivr GitHub mirror connectivity"}

[tool • pwsh]
=== https://cdn.jsdelivr.net/gh/mk-minchul/AdaFace@master/README.md
ERR 操作超时。
=== https://cdn.jsdelivr.net/gh/tinyvision/SOLIDER@main/README.md
ERR 操作超时。
=== https://cdn.jsdelivr.net/gh/Syliz517/CLIP-ReID@main/README.md
ERR 操作超时。
