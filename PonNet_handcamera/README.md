# PonNet Pytorch実装+ハンドカメラを追加したPonNet
このリポジトリには
*Predicting and Attending to Damaging Collisions for Placing Everyday Objects in Photo-Realistic Simulations*に使用されたPonNet Keras実装を元にPytorchで実装したコードが含まれます．

## 引用
詳しくは論文を参照してください:
* A. Magassouba, K.Sugiura, A.Nakayama, T.Hirakawa, T. Yamashita, H. Fujiyoshi, H. Kawai, "Predicting and Attending to Damaging Collisions for Placing Everyday Objects in Photo-Realistic Simulations," 2020.
* 河合竹彦，平川翼，山下隆義, 藤吉弘亘, 杉浦孔明,“配置物体
を考慮した Attention 機構マルチモーダルネットワークによ
る衝突予測”, 日本ロボット学会学術講演会, 2021.
## 動作環境
- Python3系
- PyTorch:1.4.0
- PyTorch vision:0.5.0
- 詳細はrequirements.txtに記載


## 使用方法
### 学習
> python3　train_handcam_end2end.py
> 
target_taskは衝突タスクの種類を選択しています．(0,1,2,3,4）の5種類で4がすべての衝突の有無に対してです．
1エポック学習するごとに，モデルの評価をします．


### ネットワークモデル
* RSJ2021のハンドカメラを用いて把持物体を考慮したマルチタスクモーダルネットワークに使用したプログラム
  * ponnetmodel_handcam_end2end_NOmeta.py
* 論文内で比較したPonNet（ベースライン）
  * ponnetmodel_baseline.py
  * ponnetmodel_baseline_Nometa.py　（把持した物体のモデル情報（メタ情報，大きさ）無しのバージョン）
* ベースネットワークResnet-18を分割して上記のモデルにinportして使用
  * resnet_pre.py
  * resnet_post.py
* それ以外のponnetmodelは実験用です．

### データーローダ
* ponnet_dataloader_handcamv2.py
* それ以外のponnet_dataloaderは実験用です．

データセットのパスを指定してください
ハンドカメラを追加したデータセットを使います．

### アテンションマップの可視化
> Python3 visualizing_attentionmap.py

可視化したいモデルのパスを指定して実行することで，RGB,Depth,handcameraのアテンションマップを可視化できます．
Raw，attention map,合成画像の3種類がそれぞれ保存されます．

### Insertion
インサーションはAttention mapを閾値でマスクし，重要な領域にアテンションが発生しているか調査する実験です．
> Python3 test_handcam_model_insertion.py

