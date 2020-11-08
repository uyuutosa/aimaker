# 学習済みAIの利用

ここでは、学習済みのAIを用いて、動画補間や超解像を行う方法を
説明します。

## 学習済みモデルの用意

適当なディレクトリ上で、下記のコマンドを実行して、各種アプリが入った`test_interp`ディレクトリを作成します。

`gen_project_for_aimaker test_interp`

`cd test_interp`でディレクトリに入ります。

`predict.py`が学習済みモデルを使って推論するためのスクリプトです。下記のように`-h`をつけて実行すると、使い方を確認することができます。

```
python predict.py -h
```

```
usage: predict.py [-h] [-o [OUTPUT]] [-d [DIVISION]] [-m [MODE]]
                  path predictor

Predict interpolation for movie

positional arguments:
  path                  feeding movie for intepolate. it can be concatenated
                        separated by comma
  predictor             interp(interpolation)|sr(super resolution) are
                        supported

optional arguments:
  -h, --help            show this help message and exit
  -o [OUTPUT], --output [OUTPUT]
                        output path
  -d [DIVISION], --division [DIVISION]
                        number of division of image
  -m [MODE], --mode [MODE]
                        interpolation : default, slash, double
                        super resolution: default, concat
```

学習済みモデルはこちらにあります。`get_model.py`スクリプトにより、学習済みモデルをダウンロードできます。

```
python get_model.py deep_voxel_flow
```

指定のパスに`deep_voxel_flow`ディレクトリが作成され、その中に`pth`ファイルおよび`ini`ファイルがダウンロードされていること確認します。

## 動画ファイルの補間

適当な動画ファイルを用意します。ここでは`movie.mp4`という動画ファイルを`test_interp`ディレクトリ直下に配置します。

```
python predict.py movie.mp4 interp
```

起動すると、補間された動画が再生されます。

### 描画モードの指定

補間前と補間後の動画を比較したいときは、`-m double`オプションを追加します。

その他には、補間前と保間を斜線で分割して比較する`-m slash`オプションも指定することができます。

### 画像の分割
AIにインプットされる動画は、設定ファイルの`dataset settings`セクションの`inputTransform`
に設定された通りに変換されます。
例えば、

```
inputTransform = resize256x256_toTensor_normalize0.5,0.5,0.5,0.5,0.5,0.5
```

の場合は256x256サイズにリサイズの上、輝度値を規格化する処理が行われます。
高解像度の画像は大きくしたほうが良いのですが、GPUのVRAMに載り切らず
メモリエラーになる場合があります。ただそのまま高解像動画を256x256サイズにリサイズすると
画質が劣化してしまうので、画像を分割したものをリサイズし、AIで推論した結果をつなぎ合わせる
ようにします。分割には`-d`オプションを使います。例えば、`-d 2`と入力すると、
縦横それぞれ2分割されます。

### 入力ソース

入力ソースは、mp4,aviの動画、画像,動画が入ったディレクトリ、カメラが現在サポートされています。
また、これらのソースをカンマ(,)でつなげることができます。つなげた場合、それらは単一のデータセットクラス
で管理されます。

カメラを入力ソースに指定する場合はid番号を入力します。例えば、0番カメラを利用するときは、以下のように
入力します。

```
python predict.py 0 interp
```

### 出力ファイル

出力形式はmp4,aviの動画、指定された名前がつけられたディレクトリに格納された画像群をサポートしています。
出力ファイルの指定は`-o`オプションで指定することができます。
例えば、`output.mp4`という動画形式で出力される場合は

```
python predict.py movie.mp4 interp -o output.mp4
```

とコマンドを実行します。同様に、`output`というディレクトリに格納された画像群で出力したい場合は、
下記のように実行します。

```
python predict.py movie.mp4 interp -o output
```

上記のように、動画か画像群かは拡張子の有無で判断されます。
