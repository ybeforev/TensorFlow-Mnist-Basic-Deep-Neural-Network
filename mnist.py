#coding : utf-8

#mnist(手書き数字データ)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#tensorflow
import tensorflow as tf

#28*28の入力画像を入力層のノードにする
#placeholder:プレースホルダ:文字列の一部を他の文字列に置換する代替物
#ニューラルネットワークの入力層のノードを用意しているのだと思う
x = tf.placeholder(tf.float32, [None, 784])

#重み変数
#Variable:変数 要素0の２次元配列[784,10]
W = tf.Variable(tf.zeros([784, 10]))

#バイアス項
b = tf.Variable(tf.zeros([10]))

#単層ニューラルネットワーク
#活性化関数にソフトマックス関数を用いる
#nn.softmax:ソフトマックス関数
y = tf.nn.softmax(tf.matmul(x, W) + b)

#ニューラルネットワークの出力層のノードを用意しているのだと思う
y_ = tf.placeholder(tf.float32, [None, 10])

#交差エントロピー誤差関数(クラス分類に用いられる誤差関数,活性化関数がソフトマックス関数(シグモイド関数)に対応)
#reduce_mean,reduce_sum:自動微分
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y), reduction_indices=[1]))

#誤差関数が小さくなるよう逆伝播公式みたいなもん
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#すべての変数を初期化
init = tf.initialize_all_variables()

#session:計算可能になったNode(edgeから送られてくる計算結果がすべてそろったNode)を非同期/並列に計算していく
sess = tf.Session()

#すべての変数を初期化
sess.run(init)

#1000個のデータで重み変数を求めている
for i in range(1000):
  #100データごとのバッチ処理を1000回している
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#argax(y,1):y配列[10]の中で最も大きい値のインデックスを返す
#argax(y,1):予測した数字
#argax(y_,1):正しいラベル
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#reduce_mean:ただの平均
#cast:自分のモデルと実データの差異に対応するコスト関数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#正解率(accuracy)を表示する
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
