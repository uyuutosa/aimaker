# aimaker

汎用的なディープラーニングの学習環境

pythonを使っていますが、設定ファイルですべて設定するだけで実行できるようになっているので、実行にあたって特にpythonの知識は必要ありません。



## Getting Started

ここでは、インストール方法と適当なアルゴリズムの学習を例示しています。
その他のチュートリアルは以下のLinkにあります。

### Link

- 動画補間AIの学習 <- here
- [超解像AIの学習](doc/train_of_superresolution_AI.md)
- [学習済みAIの利用](doc/prediction_with_trained_AI.md)
- [ソケット通信を用いたAIの遠隔利用](doc/socket_connection.md)

### Prerequisites
以下のコマンドで必要なファイルをインストールしてください。
```
pip install -r requiremtns.txt
```

### Instlling
aimakerのディレクトリの直下で、下記のコマンドを入力し環境変数を追加します。
```
./path_exportor_for_aimaker ~/.bashrc  #bashrcの部分は適宜変更してください。
```

以下のようにｒｃファイルに追記されます。
```
# aimaker settings
export PYTHONPATH=$PYTHONPATH:/path/to/pre/aimaker/dir
export AIMAKER_PATH=/path/to/aimaker/dir
export AIMAKER_APPS_PATH=$AIMAKER_PATH/apps
export PATH=$PATH:$AIMAKER_PATH/bin
```


## Running and tests

基本的にすべての設定は設定ファイルに記載してあります。タスクごとに設定ファイルや学習パラメータが生成されますので、新しくディレクトリを作り、その中で作業をします。以下の例では、`test_dir`というディレクトリを作ります。ディレクトリの作成及び、学習や推論のためのスクリプトをその中にコピーするために、以下のスクリプトを実行します。

```
gen_project_for_aimaker test_dir
```

これで、`test_dir`以下に下記のファイルが作成されます。

`cd`コマンドで`test_dir`に入ります
```
cd test_dir
```

### train
#### 設定ファイルの入力
設定ファイルは`setting.ini`です。
今回は[Deep Voxcel Flow ](https://github.com/liuziwei7/voxel-flow)を学習させてみます。このアルゴリズムは動画間の補間を学習することができます。

エディタで設定ファイルを編集します。
```
vi setting.ini
```

現状、設定ファイルは雑多になっています。今後、階層化、設定ファイルの分離をすることで整理する予定です。

|セクション名|説明|
|--------|------|
|global settings|全体的な設定|
|layer settings|学習器のレイヤーに共通の設定|
|GAN loss settings|GAN lossの設定|
|loss settings|lossに共通な設定|
|feature loss settings|vgg19を使ったlossの設定|

Deep Voxcel Flowを実装するためには、まず学習モデルや学習の方法、推論の仕方などを扱うコントローラを設定します。Deep Voxcel Flowに対応するコントローラは`voxel`なので、いかのように`global settings`に設定します。

```
controller = voxel
```

使用するgpuのidを指定します。後述のバッチの並列化を使用しない場合は、先頭のidのgpuのみ使用されます。

```
gpu_ids = 0,1,2
```

Deep Voxcel Flowは入力データが時刻tの3チャネルのカラー画像+時刻t+1の3チャネルのカラー画像の合計6チャネルになります。以下のように設定します。

```
numberofInputImageChannels = 6
```

アウトプットのチャネルは3チャネルのカラー画像になるので、以下のように設定します。

```
numberofOutputImageChannels = 3
```

バッチサイズを指定します。

```
batchsize = 1
```

バッチの並列化を行うと、バッチを各gpuに分散させることができ、バッチサイズを大きくすることができます。今回は使わないことにします。以下のように設定します。

```
isDataParallel = false
```

学習の経過は[visdom](https://github.com/facebookresearch/visdom)で知ることができます。visdomは簡易的な画像やグラフのビューアであり、webブラウザ上に表示することができます。ここでは、そのためのport番号を指定します。

```
port = 8097
```

次に、Deep voxel flow コントローラの設定をします。
`voxel flow settings`セクションの各パラメータを上記の要領で設定していきます。設定するパラメータとその値はそれぞれ以下のようになります。

|パラメータ名|値|説明|
|---------|---------|---------|
|generatorModel|UnetGenerator|U-Netで実装されたジェネレータ|
|generatorCriterion|mix|複数のLoss関数の組み合わせ(ジェネレータ用)|
|discriminatorModel|globallocal|大域と局所を組み合わせたディスクリミネータ|
|discriminatorCriterion|GANLoss|GAN用のLoss(ディスクリミネータ)|


以下のようになります。

```
[voxel flow settings]
generatorModel = unet
generatorCriterion = mix
generatorOptimizer = adam
discriminatorModel = globallocal
discriminatorCriterion = GANLoss
discriminatorOptimizer = adam
lambda = 10
imagePoolSize = 50
lambda1 = 0.0001
lambda2 = 0.005
```

U-Netをジェネレータに指定したので、その詳細を下記のように設定します。

```
[UnetForGenerator settings]
numComvolutionEachHierarchy = 4
numHierarchy = 3
scaleRatio = 2
featureSize = 64
normalizeLayer = batch
innerActivation = relu
outputActivation = tanh
```

globallocalをディスクリミネータに指定したので、その詳細を下記のように設定します。

```
[global local model settings]
numberOfInputImageChannels = 3
last activation = sigmoid
```

ジェネレータのロス関数にはmixを設定しました。この設定は`mix loss settings`セクションにて行います。以下のように設定します。

```
[mix loss settings]
lossNames = L1,feature
alphas = 1,0.01
```

`lossNames`に組み合わせるロス関数を指定します。上記ではL1ロスとネットワークをロス関数として使用するfeatureロスを指定しています。featureロスは更に設定があるので、`feature loss settings`セクションに下記のように設定します。

```
[feature loss settings]
contentCriterion = MSE
styleCriterion = MSE
alpha = 10
beta = 10
contentLossLayers = conv_4_2
styleLossLayers = conv_1_1,conv_2_1,conv_3_1,conv_4_1,conv_5_1
featureNet = vgg19
```

ジェネレータとディスクリミネータのオプティマイザにはAdamを使用しています。
Adamの設定は'optimizer settings'にて行います。
以下のように設定します。

```
[optimizer settings]
lr = 0.0001
betas = 0.9,0.999
eps = 1e-8
weightDecay = 0
```

次にデータ・セットの設定をします。
今回は動画間の補間を学ぶために、動画から隣接するフレーム2枚をフィードするデータセットを作成します。
設定は`dataset settings`セクションに以下のように行います

```
[dataset settings]
datasetName = interp
isTrain = true
isTest = false
isValid = false
interpPath = /media/yu/disk2/Dropbox/datasets/movie/UCF101
inputTransform = resize256x256_toTensor_normalize0.5,0.5,0.5,0.5,0.5,0.5
targetTransform = resize256x256_toTensor_normalize0.5,0.5,0.5,0.5,0.5,0.5
cropper = colorJitter1,1,1,0.5
isShuffle = true
trainTestRatio = 1
testValidRatio = 0
isDualOutput = false
offsetInput = 2
offsetTarget = 1
```

`datasetName`に`interp`を指定します。これにより、このデータセットは隣接フレームを
2枚ずつ出力します。
データセットは訓練データ、テストデータ、バリデーションデータに分けることができます。
1エポックごとに訓練データによる訓練が終わった後、テスト及びバリデーションを行うことができます(テストとバリデーション時には学習されません)。
これらはloss関数のグラフとしてvisdom上のグラフに表示されます。
今回はバリデーションは訓練データのドメインとは別のもので行うので、テストとバリデーションは行わないことにします。
そのため、訓練と{テスト+バリデーション}の割合`trainTestRatio`を1に、
テストとバリデーションの割合`testValidRatio`を0に指定,
`isTrain`, `isTest`, `isValid`をそれぞれ`true`, `false`, `false`にします。

データのソースには、画像or動画が入ったディレクトリ、動画ファイル、画像1枚(訓練時にはおそらく使わない)などを指定することができます。
またそれらを連結して一つのデータセットとすることができます。
今回はDeep voxel flowの論文に記載の[UCF101](http://crcv.ucf.edu/data/UCF101.php)を用いることにします。
[こちら](http://crcv.ucf.edu/data/UCF101/UCF101.rar)からrarファイルをダウンロードできます。
適当な場所に展開すると、以下のようにディレクトリ直下にたくさんのaviファイルが格納されている構造になっています。

```
UCF101
├── v_ApplyEyeMakeup_g01_c01.avi
├── v_ApplyEyeMakeup_g01_c02.avi
├── v_ApplyEyeMakeup_g01_c03.avi
:
:
├── v_YoYo_g25_c03.avi
├── v_YoYo_g25_c04.avi
└── v_YoYo_g25_c05.avi

0 directories, 13320 files
```

このデータセットを学習に使用するには、ルートのディレクトリ(UCF101)のパスを`interpPath`に指定します。

データセットは、インプットとターゲットの組になっています。それぞれに適切名前処理を施す必要があります。
今回は画像にジッターをかけた後、256x256にリサイズして、更にノーマライズをするという処理を施します。
画像のジッターはランダムに与えられますが、一つのインプットとターゲットのデータの組で同じパラメータ
である必要があります。このような共通の処理は`cropper`(`commonTransform`に変更予定)に記載します。
入力と出力の画像の前処理は`inputTransform`及び`targetTransform`にて行います。

```
cropper = colorJitter1,1,1,0.5
```

残りの処理をそれぞれのパラメータに記載します(今回は共通の処理なので、`cropper`に書いても良い)

```
inputTransform = resize256x256_toTensor_normalize0.5,0.5,0.5,0.5,0.5,0.5
targetTransform = resize256x256_toTensor_normalize0.5,0.5,0.5,0.5,0.5,0.5
```

これで必要な設定が全て終わりました。
訓練の前に、学習状況確認するためのwebサーバーを立ち上げます。

ターミナルを開いて(ディレクトリはどこでも良い)、以下のコマンドを実行します

```
python -m visdom.server
```

これでwebサーバが立ち上がり、ローカルホストのポート8097番にてvisdomにアクセスすることができるようになりました。
port番号が809７番以外の場合、例えば9000番の場合は以下のように書きます。

```
python -m visdom.server -port 9000
```

訓練は以下のコマンドで行います。

```
python train.py
```

実行すると、コマンドラインにはネットワーク情報などの設定(その他今後追加予定)及び学習の進捗状況が、ブラウザ上には設定ファイルで指定したグラフと画像がそれぞれ表示されます。

![train](doc/images/train.png)
![train](doc/images/train_visdom.png)

学習のエポック数はtrain.pyで指定できますが、Ctrl-Cで打ち切ることもできます。
打ち切った場合、

```
KeyboardInterrupt
done save models
```

のような表示が出て、設定ファイルで指定したディレクトリ(`checkPointsDirectory`にて設定)
にモデルパラメータが保存されます。再度学習あるいは推論に使う場合はこれらのモデルパラメータが
コントローラに読み込まれます。
