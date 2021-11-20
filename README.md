## PHOTO CLASSIFY BLOG

#### アプリケーションの概要

- アップロードした写真をAIで自動で画像分類する写真ブログアプリ
- 「新規投稿」、又は、既存の記事の「更新」において、記事登録時に「自動分類」を選択すると、Tensorflowを用いた画像認識モデルにより、下記の 10 カテゴリに自動で分類を行います。
- (カテゴリ一覧：Animal(動物), Architecture(建築), Art(アート), Cat(ねこ), Dessert(スイーツ), Dog(イヌ), Flower(花), Food(グルメ), Landscape(風景), People(人物))
- (なお、ポートフォリオサイトとしてご覧いただく場合、「ゲストユーザーとしてログイン」をクリックすることでゲストとしてログインでき、投稿等全ての機能を使用いただけます。)

#### 使用技術等

- Python
- Tensorflow(2.1.0)
- Django(3.1)
- PostgresSQL
- AWS(EC2)
- Let's Encrypt
- Git, Bitbucket

#### 作成にあたっての概要等

- 分類モデルは、VGG16の学習済みモデルを基に、独自にスクレイピングを行った画像(10項目×各400枚)を用いて転移学習を実施し作成(モデル作成時のevaluate：accuracy 0.8931)。
- AWSのEC2インスタンス上に、上記分類モデルを搭載したDjangoアプリを構成(Nginx, Gunicorn, PostgresSQLを使用)。独自ドメインを取得し、Let's_encrypt(Certbot)によりssl化を実施。
