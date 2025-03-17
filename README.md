# IsoSEL: Isometric Structural Entropy Learning for Deep Graph Clustering in Hyperbolic Space

The Extension of [ICML2024 (**Oral**)]: LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering.

## Get Started
Firstly, install all the required pakages and ```Python==3.9.0```.
```shell
cd ./DSE_clustering
pip install -r requirements.txt
```

Run ```main.py``` to train and test model.
```shell
python main.py
```
You can design your own configurations in ```configs``` directory by ```.json``` files.

Pay attention to the following codes to load or save your configurations:
```python
parser = argparse.ArgumentParser(description='Lorentz Structural Entropy')
# ...add_argument
configs = parser.parse_args()

# save to json file
with open(f'./configs/{configs.dataset}.json', 'wt') as f:
    json.dump(vars(configs), f, indent=4)

# load from json file
configs_dict = vars(configs)
with open(f'./configs/{configs.dataset}.json', 'rt') as f:
    configs_dict.update(json.load(f))
configs = DotDict(configs_dict)

f.close()

```

## Visualization

<div align=center>
<img src="./images/FootBall_pred.png" width=50% alt="football" title="FootBall" >
</div>
<div align=center>
Figure 1. Prediction results on FootBall dataset.
</div>
<br><br>
<div align=center>
<img src="./images/FootBall_true.png" width=50% alt="football" title="FootBall">
</div>
<div align=center>
Figure 2. True labels of FootBall dataset.
</div>
