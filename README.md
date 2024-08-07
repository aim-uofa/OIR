

<!-- # magic-edit.github.io -->

<p align="center">

  <h2 align="center">Object-aware Inversion and Reassembly for Image Editing</h2>
  <p align="center">
    <a href="https://zhenyangcs.github.io/"><strong>Zhen Yang*</strong></a>
    ·
    <a href="https://github.com/dingangui"><strong>Ganggui Ding*</strong></a>
    ·  
    <a href="https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl=zh-CN"><strong>Wen Wang*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=FaOqRpcAAAAJ"><strong>Hao Chen*</strong></a>
    ·
    <a href="https://bohanzhuang.github.io/"><strong>Bohan Zhuang†</strong></a>
    ·
    <a href="https://cshen.github.io/"><strong>Chunhua Shen*</strong></a>
    <br>
    *Zhejiang University&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;†Monash University
    <br>
    </br>
        <a href="https://arxiv.org/abs/2310.12149">
        <img src='https://img.shields.io/badge/Arxiv-OIR-blue' alt='Paper PDF'></a>
        <a href="https://aim-uofa.github.io/OIR-Diffusion/">
        <img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
        <a href="https://drive.google.com/file/d/1JX8w0S9PCD9Ipmo9IiICO8R7e1haTGdF/view?usp=sharing">
        <img src='https://img.shields.io/badge/Dataset-OIR--Bench-green' alt='OIR-Bench'></a>
        <a href="https://iclr.cc/virtual/2024/poster/18242">
        <img src='https://img.shields.io/badge/Video-ICLR-yellow' alt='Video'></a>
  </p>
</p>


<!-- <p align="center"><b>We will release the code soon!</b></p> -->

## Setup
This code was tested with Python 3.9, Pytorch 2.0.1 using pre-trained models through huggingface / diffusers. Specifically, we implemented our method over Stable Diffusion 1.4. Additional required packages are listed in the requirements file. The code was tested on a NVIDIA GeForce RTX 3090 but should work on other cards.

## Getting Started
1. Download OIR-Bench.
2. Create the environment and install the dependencies by running:
```
conda create -n oir python=3.9
conda activate oir
pip install -r requirements.txt
```
3. Change the **basic_config.py** in **configs/**, change the model path and hyperparameters.
4. Modify **multi_object_edit.yaml** or **single_object_edit.yaml** in **configs/** according to **multi_object.yaml** and **single_object.yaml** in **OIR-Bench/**.
5. Run **single_object_edit.py** (Search Metric in paper) or **multi_object_edit.py** (OIR in paper) to implement image editing.
6. **Option**: Adjust **reassembly_step** and repeat the above process to get better results.

## TODO
1. Use prompt_change as dict's key may lead to error.
2. Different editing pairs' masks mustn't have overlap.
3. Search metric can be an ensemble learning tool. For example, we can use pnp, p2p, OIR ... method to edit an image and we can use search metric to select the optimal editing result.
4. We can also use the method in TODO 3 to build a high quality dataset to train instruct-based image editing method.
5. Deploy our method on different foundation model (SDXL, LCM ...)

## Results

### OIR results
<p align="center">
  <table align="center">
    <td>
      <img src="./assets/OIR_result_1.png"></img>
      <!-- <img src="./assets/OIR_result_2.png"></img>
      <img src="./assets/OIR_result_3.png"></img> -->
    </td>
  </table>
</p>

### Visualization of the search metric
<p align="center">
  <table align="center">
    <td>
      <img src="./assets/search_metric_1.png"></img>
      <!-- <img src="./assets/search_metric_2.png"></img> -->
    </td>
  </table>
</p>



## Acknowlegment
Many thanks for the generous help in building the project website from Minghan Li.

## Citing
If you find our work useful, please consider citing:


```BibTeX
@article{yang2023OIR,
  title={Object-aware Inversion and Reassembly for Image Editing},
  author={Yang, Zhen and Ding, Ganggui and Wang, Wen and Chen, Hao and Zhuang, Bohan and Shen, Chunhua},
  publisher={arXiv preprint arXiv:2310.12149},
  year={2023},
}
```

