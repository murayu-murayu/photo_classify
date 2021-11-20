from django.db import models
from django.contrib.auth.models import AbstractUser

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64



class CustomUser(AbstractUser):
    """拡張ユーザーモデル"""
       
    class Meta:
        verbose_name_plural = 'CustomUser'


class Category(models.Model):
    name = models.CharField("カテゴリ名", max_length=50)
    name_en = models.CharField("カテゴリ名英語", max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Post(models.Model):
    author = models.ForeignKey("CustomUser", on_delete=models.PROTECT, blank=False)
    title = models.CharField("タイトル", max_length=50)
    content = models.TextField("内容", max_length=1000)
    category = models.ForeignKey("Category", on_delete=models.PROTECT)
    thumbnail = models.ImageField(upload_to="images/", blank=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class Like(models.Model):
    post = models.ForeignKey(Post, verbose_name="投稿", on_delete=models.PROTECT)
    user = models.ForeignKey(CustomUser, verbose_name="Likeしたユーザー", on_delete=models.PROTECT)


graph = tf1.get_default_graph()


class Photo(models.Model):
    image = models.ImageField(upload_to='images/')

    IMAGE_SIZE = 224 # 画像サイズ
    MODEL_FILE_PATH = './classify/ml_models/vgg16_transfer.h5' # モデルファイル
    classes = ["2", "3", "4", "5", "6", "7", 
              "8", "9", "10", "11"]
    num_classes = len(classes)

    # 引数から画像ファイルを参照して読み込む
    def predict(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)
            
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image) / 255.0
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return self.classes[predicted], percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' + img.file.content_type + ';base64,' + base64_img
