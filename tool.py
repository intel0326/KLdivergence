#coding: utf-8
#windows環境であれば「#coding: cp932」に変更

import numpy as np


#ガウス分布を仮定した平均と共分散を求める
def estimation( x ):

    #x=(20, 1000)
    #20次元の特徴量、1000個

    #平均 xMean=(20, )
    xMean = np.mean(x, axis=0)
    #共分散 xCov=(20, 20)
    xCov = np.cov(x, rowvar=0, bias=1)

    return xMean, xCov


#KL情報量をもとめる
#xMean1, xCov1は一つ目の分布、xMean2, xCovは二つ目の分布
def KL( xMean1, xCov1, xMean2, xCov2 ):

    #多次元ガウス分布の次元dを算出する
    d = np.trace( np.dot( np.linalg.inv( xCov1 ) ), xCov1 )

    #共分散行列の行列式を算出
    xCov1_det = np.linalg.det(xCov1)
    xCov2_det = np.linalg.det(xCov2)
    L = np.log( xCov1_det / xCov2_det )

    #共分散行列のトレース
    T = np.trace( np.dot( np.dot( np.linalg.inv( xCov2 ), xCov1 ) ) )

    #平均の内積
    M = np.dot( ( xMean1 - xMean2 ), np.linalg.inv( xCov2 ) )
    M = np.dot( M, ( xMean1 - xMean2 ) )

    #KL情報量
    y = ( 1 / 2 ) * ( ( -1 ) * L + T + M - d )

    return y

