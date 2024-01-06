# Federated-Learning (PyTorch)

Implementation of the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.


### Results
epoch: 10
Dataset: MNIST
frac: 0.1
Non-IID (equal): NIE
| Model | IID-Atte | IID-Att2| IID-Att5|IID-Att8|IID-Att10 |IID-11|IID-Att12| IID-Att14 |IID-Att15 |IID-Att16| IID-Att17|IID-Avg |IID-Att18|IID-Att19|20|
| ----- | -----    | ---      |---       |---      |---       |---   |---       |---        |---       |---      |---  |---   |---|---|---|
|  MLP  |     -    |          |          |         |          |      |          |           |          |         |        | || |
|  CNN  |  96.93%  |  97.10%  |97.00%    | 95.80%  |**97.15%**|97.10%|96.25%    |97.12%     |97.12%    |97.12%   | 97.11%|97.01%|95.73%|95.57%|97.03%|

| Model |NIE-Atte  |NIE-Att2 |NIE-Att5 |NIE-Att8|NIE-10 | NIE-11  |NIE-Att12| NIE-att14|NIE-att15 |NIE-att16|NIE-att17|NIE-Avg |NIE-att18|NIE-Att18|20|
| ----- |----      | ---      |---       | ---     |---    |---      |---       | ---      |---       |---      | ---    | --- |---|---|---|
|  MLP  |          |    -     |          |         |       |         |          |          |          |         |        ||||
|  CNN  | 81.54%   | 82.62%   |  82.63%  | 75.86%  | 81.17%| 81.53%  | 71.77%   |**82.89%**|  82.45%  |82.31%   |81.81%|82.48%| 81.28%|68.34%|80.91%|


==轮次太少不具备参考价值==

epoch: 10
Dataset: Cifar10
frac: 0.1
Non-IID (equal): NIE
| Model | IID-Atte |IID-Atte2 |IID-Atte5 |IID-Atte16|IID-Avg  | 
| ----- | -----    |---       |---       |---       |---      |
|  MLP  |     -    |          |          |          |         |
|  CNN  |  45.84%  | 46.15%   |  46.47%  |**46.63%**|46.21%   |
| Model |NIE-Atte  |NIE-Atte2 |NIE-Atte3 |NIE-Atte5 |IID-Atte16|NIE-Avg  |
| ----- |----      |---       |---       |---       |---       | ---     |
|  MLP  |    -     |          |          |          |          |         |
|  CNN  |**28.28%**| 26.44%   |  26.65%  |26.44%    |  23.21   | 27.19%  |



Cifar10
epoch: 10
frac: 0.3
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
.tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
.tg-sort-header::-moz-selection{background:0 0}
.tg-sort-header::selection{background:0 0}.tg-sort-header{cursor:pointer}
.tg-sort-header:after{content:'';float:right;margin-top:7px;border-width:0 5px 5px;border-style:solid;
  border-color:#404040 transparent;visibility:hidden}
.tg-sort-header:hover:after{visibility:visible}
.tg-sort-asc:after,.tg-sort-asc:hover:after,.tg-sort-desc:after{visibility:visible;opacity:.4}
.tg-sort-desc:after{border-bottom:none;border-width:5px 5px 0}</style>
<table id="tg-sS5Jk" class="tg">
<thead>
  <tr>
    <th class="tg-8jgo">Dataset</th>
    <th class="tg-8jgo" colspan="34">Cifar</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8jgo">Data</td>
    <td class="tg-8jgo" colspan="17">IID</td>
    <td class="tg-8jgo" colspan="17">Non-IID</td>
  </tr>
  <tr>
    <td class="tg-8jgo">aggregation</td>
    <td class="tg-8jgo">avg</td>
    <td class="tg-8jgo">att1</td>
    <td class="tg-8jgo">att2</td>
    <td class="tg-8jgo">att3</td>
    <td class="tg-8jgo">att4</td>
    <td class="tg-8jgo">att5</td>
    <td class="tg-8jgo">att6</td>
    <td class="tg-8jgo">att7</td>
    <td class="tg-8jgo">att8</td>
    <td class="tg-8jgo">att9</td>
    <td class="tg-8jgo">att10</td>
    <td class="tg-8jgo">att11</td>
    <td class="tg-8jgo">att12</td>
    <td class="tg-8jgo">att13</td>
    <td class="tg-8jgo">att14<br></td>
    <td class="tg-zv4m">att15</td>
    <td class="tg-zv4m">att16</td>
    <td class="tg-8jgo">avg</td>
    <td class="tg-8jgo">att1</td>
    <td class="tg-8jgo">att2</td>
    <td class="tg-8jgo">att3</td>
    <td class="tg-8jgo">att4</td>
    <td class="tg-8jgo">att5</td>
    <td class="tg-8jgo">att6</td>
    <td class="tg-8jgo">att7</td>
    <td class="tg-8jgo">att8</td>
    <td class="tg-8jgo">att9</td>
    <td class="tg-8jgo">att10</td>
    <td class="tg-8jgo">att11</td>
    <td class="tg-8jgo">att12</td>
    <td class="tg-8jgo">att13</td>
    <td class="tg-8jgo">att14</td>
    <td class="tg-zv4m">att15</td>
    <td class="tg-zv4m">att16</td>
  </tr>
  <tr>
    <td class="tg-8jgo">MLP</td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-zv4m"></td>
    <td class="tg-zv4m"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-zv4m"></td>
    <td class="tg-zv4m"></td>
  </tr>
  <tr>
    <td class="tg-8jgo">CNN</td>
    <td class="tg-8jgo">47.24%</td>
    <td class="tg-8jgo">47.69%</td>
    <td class="tg-8jgo">47.33%</td>
    <td class="tg-8jgo">47.31%</td>
    <td class="tg-8jgo">46.83%</td>
    <td class="tg-8jgo">47.47%</td>
    <td class="tg-8jgo">47.65%</td>
    <td class="tg-8jgo">47.12%</td>
    <td class="tg-8jgo">47.28%</td>
    <td class="tg-8jgo">46.87%</td>
    <td class="tg-8jgo">47.29%</td>
    <td class="tg-8jgo">47.37%</td>
    <td class="tg-8jgo">46.47%</td>
    <td class="tg-8jgo">47.03%</td>
    <td class="tg-8jgo">47.33%</td>
    <td class="tg-zv4m">47.47%</td>
    <td class="tg-zv4m">47.40</td>
    <td class="tg-8jgo">32.15%</td>
    <td class="tg-8jgo">33.35%</td>
    <td class="tg-8jgo">32.19%</td>
    <td class="tg-8jgo">32.09%</td>
    <td class="tg-8jgo">32.24%</td>
    <td class="tg-8jgo">32.23%</td>
    <td class="tg-8jgo">31.37%</td>
    <td class="tg-8jgo">32.57%</td>
    <td class="tg-8jgo">33.87%</td>
    <td class="tg-8jgo">32.42%</td>
    <td class="tg-8jgo">33.43%</td>
    <td class="tg-8jgo">33.40%</td>
    <td class="tg-8jgo">26.49%</td>
    <td class="tg-8jgo">32.28%</td>
    <td class="tg-8jgo">32.63%</td>
    <td class="tg-zv4m">32.49%</td>
    <td class="tg-zv4m">32.95%</td>
  </tr>
</tbody>
</table>
<script charset="utf-8">var TGSort=window.TGSort||function(n){"use strict";function r(n){return n?n.length:0}function t(n,t,e,o=0){for(e=r(n);o<e;++o)t(n[o],o)}function e(n){return n.split("").reverse().join("")}function o(n){var e=n[0];return t(n,function(n){for(;!n.startsWith(e);)e=e.substring(0,r(e)-1)}),r(e)}function u(n,r,e=[]){return t(n,function(n){r(n)&&e.push(n)}),e}var a=parseFloat;function i(n,r){return function(t){var e="";return t.replace(n,function(n,t,o){return e=t.replace(r,"")+"."+(o||"").substring(1)}),a(e)}}var s=i(/^(?:\s*)([+-]?(?:\d+)(?:,\d{3})*)(\.\d*)?$/g,/,/g),c=i(/^(?:\s*)([+-]?(?:\d+)(?:\.\d{3})*)(,\d*)?$/g,/\./g);function f(n){var t=a(n);return!isNaN(t)&&r(""+t)+1>=r(n)?t:NaN}function d(n){var e=[],o=n;return t([f,s,c],function(u){var a=[],i=[];t(n,function(n,r){r=u(n),a.push(r),r||i.push(n)}),r(i)<r(o)&&(o=i,e=a)}),r(u(o,function(n){return n==o[0]}))==r(o)?e:[]}function v(n){if("TABLE"==n.nodeName){for(var a=function(r){var e,o,u=[],a=[];return function n(r,e){e(r),t(r.childNodes,function(r){n(r,e)})}(n,function(n){"TR"==(o=n.nodeName)?(e=[],u.push(e),a.push(n)):"TD"!=o&&"TH"!=o||e.push(n)}),[u,a]}(),i=a[0],s=a[1],c=r(i),f=c>1&&r(i[0])<r(i[1])?1:0,v=f+1,p=i[f],h=r(p),l=[],g=[],N=[],m=v;m<c;++m){for(var T=0;T<h;++T){r(g)<h&&g.push([]);var C=i[m][T],L=C.textContent||C.innerText||"";g[T].push(L.trim())}N.push(m-v)}t(p,function(n,t){l[t]=0;var a=n.classList;a.add("tg-sort-header"),n.addEventListener("click",function(){var n=l[t];!function(){for(var n=0;n<h;++n){var r=p[n].classList;r.remove("tg-sort-asc"),r.remove("tg-sort-desc"),l[n]=0}}(),(n=1==n?-1:+!n)&&a.add(n>0?"tg-sort-asc":"tg-sort-desc"),l[t]=n;var i,f=g[t],m=function(r,t){return n*f[r].localeCompare(f[t])||n*(r-t)},T=function(n){var t=d(n);if(!r(t)){var u=o(n),a=o(n.map(e));t=d(n.map(function(n){return n.substring(u,r(n)-a)}))}return t}(f);(r(T)||r(T=r(u(i=f.map(Date.parse),isNaN))?[]:i))&&(m=function(r,t){var e=T[r],o=T[t],u=isNaN(e),a=isNaN(o);return u&&a?0:u?-n:a?n:e>o?n:e<o?-n:n*(r-t)});var C,L=N.slice();L.sort(m);for(var E=v;E<c;++E)(C=s[E].parentNode).removeChild(s[E]);for(E=v;E<c;++E)C.appendChild(s[v+L[E-v]])})})}}n.addEventListener("DOMContentLoaded",function(){for(var t=n.getElementsByClassName("tg"),e=0;e<r(t);++e)try{v(t[e])}catch(n){}})}(document)</script>
接上表格
NIE-att17: 33.26%
IID-att17: 47.66%
NIE-att18: 32.93%
IID-att18: 47.13%
NIE-att19: 26.82%
IID-att19: 43.51%
IID-att20: 47.18%(10rounds不收敛)48.44%
NIE-att20: 31.35%(10rounds不收敛)41.71%

epoch: 50
Dataset: Cifar10
frac: 0.1
Non-IID (equal): NIE
| Model |  IID-Avg | IID-Atte | IID-Atte2 |IID-Atte3 |IID-Atte4|IID-Atte5  |IID-att11 |IID-att14 |IID-Atte15|IID-Atte16|IID-Atte17|IID-Att18|IID-Att19|
| ----- | ---      | -----    | ---       |---       |---      |---        |---       |---       |---       |---       |---| ---|---|
|  MLP  |          | -        |           |          |         |           |          |          |          |          | | ||
|  CNN  |  48.52%  |  48.34%  |  49.03%   |  49.35%  |  49.18% |  49.77%   | 49.26%   |  49.89%  | 49.64%   |**50.20%**| 49.45%|49.40%|49.08%|

| Model | NIE-Avg  | NIE-Atte |NIE-Atte2  |NIE-Atte3 |NIE-Atte4|NIE-Atte5  |NIE-Atte11|NIE-Atte14|NIE-Atte15|NIE-Atte16|NIE-Atte17|NIE-Att18|NIE-Att19|
| ----- |----      |---       |---        | ---      | ---     |---        |  ---        |  ---     |---       |---       |---| --- |---|
|  MLP  |-         |          |           |          |         |           |          |          |          |          | | ||
|  CNN  | 38.28%   |38.28%    | 37.78%    |37.88%    |  37.92% | 38.20%    |  38.44%  |  37.57%  | 38.59% |41.54%|41.97%|43.74%| **43.93%**|


经过综合对比，选择14<16<17。

* To run the federated experiment with CIFAR on CNN (IID, Aggregate: Fedavg):
```
python src/federated_main.py --agg avg --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_main.py --agg avg --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```
* To run the federated experiment with CIFAR on CNN (IID, Aggregate: Attention):
```
python src/federated_main.py --agg att --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_main.py --agg att --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

* To run the federated experiment with MNIST on CNN (IID, Aggregate: Attention):
```
python src/federated_main.py --agg att --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
```

* To run the same experiment under non-IID condition:
```
python src/federated_main.py --agg att --model=cnn --dataset=mnist --gpu=0 --iid=0 --epochs=10
```

* To run the federated experiment with MNIST on CNN (IID, Aggregate: Fedavg):
```
python src/federated_main.py --agg avg --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```

* To run the same experiment under non-IID condition:
```
python src/federated_main.py --agg avg --model=cnn --dataset=mnist --gpu=0 --iid=0 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD 
* ```Learning Rate:``` 0.01
* ```seed```: 42

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.64%  |
|  CNN  |  98.23%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01


```Table 2:``` Test accuracy after training for 10 global epochs with:

| Aggregation | Model |    IID   | Non-IID (equal)|
| ---         | ----- | -----    |----            |
|Att16        |  MLP  |  91.11%  |     71.42%     |
|Att14        |  MLP  |  91.20%  |     73.23%     |
|Atte         |  MLP  |  91.22%  |     71.74%     |
|Att11        |  MLP  |          |                |
|FedAvg       |  MLP  |  91.20%  |     73.28%     |



## Further Readings
### Papers:
* [Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/pdf/1812.07108.pdf)
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
