### GRU
- [x] 1<sub>st</sub> idea about W 
- [ ] 2<sub>st</sub> idea about W
- [ ] 3<sub>st</sub> idea about W

***
### CNN
- cnn_AE_1是第一种想法，在时间维度上卷积。卷积操作中的卷积核相同。
- cnn_AE_1pro是cnn_AE_1的进阶版，卷积操作中的卷积核不一样大。反卷积操作未作改变。在卷积操作中加了高斯噪声，但被我注释掉了，按需要可注释回来。
- cnn_AE_2是第二种想法，卷积核不是常规的正方形。
- 所有卷积操作的strides都为1，后续按需会做改变。
- CNN输入样本数必须为**6**的倍数， 时间步数必须为**8**的倍数

