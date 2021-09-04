# Ganimation
## 概要
---
  
AIとPythonを使ってZoom等のコミュニケーションツールでアニメのキャラクターになりきれるアプリです。

<img src="./Images/for_readme/sample01.gif" >

## 使い方
---
[こちら](https://drive.google.com/file/d/1_LwXFVwz85g6T6CV-kqXaWQYMxo8U23n/view?usp=sharing)からdata.zipをダウンロードして解凍した後、Ganimation\Animatorフォルダに配置してください。
[こちら](https://drive.google.com/file/d/1UFVbqX8fPCEC5PTfvcyLE-uCgLPD322m/view?usp=sharing)からtemp.zipをダウンロードして解凍した後に、Ganimationフォルダに配置してください。
Screenフォルダ内の”Ganimation.bat”をクリックすると実行されます。

**画像の条件**

1. お好きなアニメ顔画像を準備します
1. [コチラ](https://www.remove.bg/ja)などで背景の透過処理を行う
1. 四角の場所にドラッグ＆ドロップしてください
1. 読み込みが完了すればアニメキャラクターになりきれます！ 
 
## Requirement
---
以下を実行すればライブラリがインストールされます
~~~
conda env create -n ganimation -f ganimation_env.yml
~~~
**推奨スペック**

||条件|
----|----
|Python|3.6|
|GPU|RTX2070super|
|メモリ|16GB|
|cuda|10以上|
|OS|Windows10|

## Licence
---
Ganimation is under [GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)

## 参考
---
AIを作るうえで以下のサイトおよびコードを参考もしくは使用しました。

[Talking Head Animefrom a Single Image](https://pkhungurn.github.io/talking-head-anime/)

[talking-head-anime-demo](https://github.com/pkhungurn/talking-head-anime-demo)


