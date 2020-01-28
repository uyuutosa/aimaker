# ソケット通信を用いたAIの遠隔利用

ここでは、ソケット通信を用いて、学習済みモデルを持つ別のPCにデータを送り推論を実行させ、推論結果を返却させることを行います。

## サーバ側(データを送信する側)の準備

下記のようにアプリケーションと設定ファイルを指定します。

```
gen_project_for_aimaker test_server
```

`cd test_server`で中に入り、下記の通り実行し設定ファイルをダウンロードします。


```
python get_model.py deep_voxel_flow ./ -only_setting
```

`-only_setting`オプションを指定することで、設定ファイルのみをダウンロードすることができます。

サーバ側は`predict_server.py`を使います。`-h`オプションで使い方を見ることができます。

```
usage: predict_server.py [-h] [-o [OUTPUT]] path predictor

server for predictor

positional arguments:
  path                  feeding movie for intepolate. it can be concatenated
                        separated by comma
  predictor             iterp(interpolation)|sr(super resolution) are
                        supported

optional arguments:
  -h, --help            show this help message and exit
  -o [OUTPUT], --output [OUTPUT]
                        output path
```

今回は例として、動画ソースは`movie.mp4`、出力は`output.mp4`とし、
AIは動画補間を使用します。
`test_server`ディレクトリ直下に`movie.mp4`があると仮定して、下記のコマンドを実行します。

```
python predict_server movie.mp4 interp -o output.mp4
```

ネットワーク情報が表示された後、以下のように表示されます。

```
waiting for connections as Server of localhost:1234...
```

## クライアント側(データを受け取り、AIで処理する側)の準備

サーバ側と同じように`gen_project_for_aimaker`でアプリケーション
を配置し、`get_model.py`でモデルをダウンロードします。

クライアント側は[学習済みAIの利用](prediction_with_trained_AI.md)
でも使用した`predict.py`を使います。
使用方法は前回とほぼ同じで、データのソースのパスの代わりにサーバのipアドレスを
入力します。例として、下記のようになります。

```
python predict.py 192.168.1.1 interp
```

クライアント側はサーバー側からデータをもらい、推論した結果をサーバーに帰すので
出力結果はクライアント側に残りません(`-o`オプションが無効になります)。

### 実行結果

コネクションが成功すると

```
```

と出力され、クライアント側で推論が行われます。この処理は[学習済みAIの利用](prediction_with_trained_AI.md)で説明した内容と同じです。

処理が終了すると、サーバー側にファイルが保存されます。この例では、`output.mp4`
が保存されます。
