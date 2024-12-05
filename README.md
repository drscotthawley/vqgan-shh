# vqgan-shh

*My lil VQGAN exercise. Uses Oxford Flowers dataset.*

This repo is not intended for public use. Just a place to store work in progress. That said, it's open and MIT-licensed, so if you find it useful, great.

## Install

```bash
pip install git+https://github.com/drscotthawley/vqgan-shh.git
```

or 

```bash
git clone https://github.com/drscotthawley/vqgan-shh.git
cd vqgan-shh
pip install -e .
```

## Run

```bash
./train-vqgan.py
```

See `./train-vqgan.py --help` for options.

## Example Results

Trained using a Razer Blade 16" with a NVIDIA 4090-mobile (16 GB VRAM) for 6 hours.  

Inputs: 128x128x3, downsampled to 32x32, with 1024 codebook vectors of 256 elements each.


### Reconstructed Images (128x128x3):
![Example image output](examples/example_demo.png)

Top row: Input images. Bottom row: Reconstructions.

### Codebook Vectors
![Example codebook vector histograms](examples/example_histogram.png)

## TODO:
- Add attention aka "non-local blocks"
- Make sure the GAN part is really helping/learning
