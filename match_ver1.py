import cv2
import numpy as np
import sys

# 1. 引数として2枚の画像のパスを取る
args = sys.argv
print('入力画像1:'+args[1])
print('入力画像2:'+args[2])

# 2. 画像を読み込む
img1 = cv2.imread(args[1])
img2 = cv2.imread(args[2])

# 3. 特徴量の抽出
## 3.1 検出ロジックの定義
##　今回はAKAZE法を使用する
print('特徴量の抽出開始')
detector = cv2.AKAZE_create()

## 3.2 1枚目の特徴量を抽出
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)

# 1枚目の画像に特徴量をマーキングした画像を出力
out = cv2.drawKeypoints(img1, keypoints1, None)
cv2.imwrite('test1_key.jpg', out)

## 3.3 2枚目の特徴量を抽出
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

# 2枚目の画像に特徴量をマーキングした画像を出力
out = cv2.drawKeypoints(img2, keypoints2, None)
cv2.imwrite('test2_key.jpg', out)

# 4.特徴量の比較
print('特徴量のマッチング開始')

## 4.1 ORBを使用した総当たりマッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

## 4.2 特徴点の距離でソートし、上位5%を抽出
matches = sorted(matches, key = lambda x:x.distance)

good = matches[:int(len(matches) * 0.05)]
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

## 4.3 射影変換を行う
h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
height, width, channels = img1.shape
dst_img = cv2.warpPerspective(img2, h, (width, height))

# 5.変換後の画像を出力
print('結果出力')
cv2.imwrite(args[2], dst_img)

# 1枚目,2枚目を水平方向に結合し、マッチした特徴点をマーキングして出力
h1, w1, c1 = img1.shape[:3]
h2, w2, c2 = img2.shape[:3]
height = max([h1,h2])
width = w1 + w2
out = np.zeros((height, width, 3), np.uint8)

cv2.drawMatches(img1,keypoints1,img2,keypoints2,good, out, flags=0)
cv2.imwrite('test_match.jpg', out)
