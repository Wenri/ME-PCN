# ME-PCN: Point Completion Conditioned on Mask Emptiness

[[paper]](https://arxiv.org/abs/2108.08187)

![](poster.png)

ME-PCN is a point completion network that leverages emptiness
in 3D shape space. It encodes both the occupied point cloud and the
neighboring ‘empty points’. It estimates coarse-grained but
complete and reasonable surface points in the first stage,
followed by a refinement stage to produce fine-grained sur-
face details.

## Install
### Envrionment & prerequisites

- PyTorch 1.10.1
- CUDA 11.3
- Python 3.9
- [Open3D](http://www.open3d.org/docs/release/index.html#python-api-index)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Visdom](https://github.com/facebookresearch/visdom)

### Compile extension modules:

```bash
cd emd
python3 setup.py install
cd expansion_penalty
python3 setup.py install
cd MDS
python3 setup.py install
```

## Usage

### Download data and trained models

We include demo model and data in demo folder. 
Those files are uploaded using [Git Large File Storage](https://git-lfs.github.com/).
To get those file inplace, you will need to install Git LFS before clone this repo.
This can be done in Debian using APT:

```bash
sudo apt install git-lfs
git lfs install
```

### Export visualization results on realdata

Unzip `realdata.zip` in `demo` folder. Run `export_realdata.py` to get the result.