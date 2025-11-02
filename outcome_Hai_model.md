Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\google\protobuf\runtime_version.py:98: UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1 at tensorflow/core/framework/cost_graph.proto. Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\google\protobuf\runtime_version.py:98: UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1 at tensorflow/core/framework/step_stats.proto. Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\google\protobuf\runtime_version.py:98: UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1 at tensorflow/core/framework/allocation_description.proto. Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\google\protobuf\runtime_version.py:98: UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1 at tensorflow/core/framework/tensor_description.proto. Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\google\protobuf\runtime_version.py:98: UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1 at tensorflow/core/protobuf/cluster.proto. Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\google\protobuf\runtime_version.py:98: UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1 at tensorflow/core/protobuf/debug.proto. Please update the gencode to avoid compatibility violations in the next runtime release.
  warnings.warn(
2025-10-31 18:30:02.389420: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Classes (7): ['background', 'cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
2025-10-31 18:30:03.840990: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0317 - loss: 1.6565 - sem_logits_loss: 1.64702025-10-31 18:41:27.544960: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.610  valmIoU=0.103  valBCE=254.5366  per-class IoU=[0.617 0.    0.    0.    0.1   0.001 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.103)
88/88 ━━━━━━━━━━━━━━━━━━━━ 684s 8s/step - boundary_logits_loss: 0.0293 - loss: 1.5379 - sem_logits_loss: 1.5292
Epoch 2/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 6s/step - boundary_logits_loss: 0.0256 - loss: 1.4579 - sem_logits_loss: 1.45022025-10-31 18:52:24.786206: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.612  valmIoU=0.106  valBCE=157.6527  per-class IoU=[0.622 0.    0.    0.014 0.084 0.022 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.106)
88/88 ━━━━━━━━━━━━━━━━━━━━ 657s 7s/step - boundary_logits_loss: 0.0257 - loss: 1.4444 - sem_logits_loss: 1.4367
Epoch 3/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 6s/step - boundary_logits_loss: 0.0246 - loss: 1.4177 - sem_logits_loss: 1.4103[Eval] valPA=0.608  valmIoU=0.117  valBCE=107.3202  per-class IoU=[0.635 0.    0.145 0.    0.    0.039 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.117)
88/88 ━━━━━━━━━━━━━━━━━━━━ 668s 8s/step - boundary_logits_loss: 0.0247 - loss: 1.4161 - sem_logits_loss: 1.4087
Epoch 4/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0241 - loss: 1.3931 - sem_logits_loss: 1.38582025-10-31 19:15:26.615721: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.612  valmIoU=0.129  valBCE=145.7183  per-class IoU=[0.636 0.    0.    0.    0.15  0.114 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.129)
88/88 ━━━━━━━━━━━━━━━━━━━━ 714s 8s/step - boundary_logits_loss: 0.0240 - loss: 1.4165 - sem_logits_loss: 1.4093
Epoch 5/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0270 - loss: 2.1904 - sem_logits_loss: 2.1823[Eval] valPA=0.611  valmIoU=0.103  valBCE=145.2128  per-class IoU=[0.621 0.045 0.    0.004 0.015 0.032 0.   ]
88/88 ━━━━━━━━━━━━━━━━━━━━ 714s 8s/step - boundary_logits_loss: 0.0245 - loss: 1.6168 - sem_logits_loss: 1.6095
Epoch 6/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0228 - loss: 1.3922 - sem_logits_loss: 1.3854[Eval] valPA=0.611  valmIoU=0.120  valBCE=110.0302  per-class IoU=[0.632 0.002 0.031 0.005 0.057 0.114 0.   ]
88/88 ━━━━━━━━━━━━━━━━━━━━ 709s 8s/step - boundary_logits_loss: 0.0222 - loss: 1.3464 - sem_logits_loss: 1.3398
Epoch 7/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0236 - loss: 1.3119 - sem_logits_loss: 1.3048[Eval] valPA=0.614  valmIoU=0.140  valBCE=118.9729  per-class IoU=[0.645 0.009 0.    0.016 0.171 0.127 0.01 ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.140)
88/88 ━━━━━━━━━━━━━━━━━━━━ 719s 8s/step - boundary_logits_loss: 0.0222 - loss: 1.3121 - sem_logits_loss: 1.3054
Epoch 8/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0208 - loss: 1.3002 - sem_logits_loss: 1.29392025-10-31 20:03:08.570853: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.613  valmIoU=0.134  valBCE=126.1329  per-class IoU=[0.634 0.056 0.055 0.07  0.058 0.066 0.   ]
88/88 ━━━━━━━━━━━━━━━━━━━━ 720s 8s/step - boundary_logits_loss: 0.0212 - loss: 1.2951 - sem_logits_loss: 1.2888
Epoch 9/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0223 - loss: 1.3105 - sem_logits_loss: 1.3038[Eval] valPA=0.600  valmIoU=0.185  valBCE=155.9759  per-class IoU=[0.652 0.028 0.145 0.1   0.079 0.154 0.139]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.185)
88/88 ━━━━━━━━━━━━━━━━━━━━ 711s 8s/step - boundary_logits_loss: 0.0217 - loss: 1.2765 - sem_logits_loss: 1.2700
Epoch 10/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0219 - loss: 1.2770 - sem_logits_loss: 1.2704[Eval] valPA=0.588  valmIoU=0.187  valBCE=178.4103  per-class IoU=[0.656 0.058 0.    0.058 0.177 0.15  0.209]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.187)
88/88 ━━━━━━━━━━━━━━━━━━━━ 725s 8s/step - boundary_logits_loss: 0.0218 - loss: 1.2837 - sem_logits_loss: 1.2771
Epoch 11/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0212 - loss: 1.2663 - sem_logits_loss: 1.2599[Eval] valPA=0.613  valmIoU=0.154  valBCE=117.4636  per-class IoU=[0.647 0.    0.008 0.052 0.195 0.156 0.022]
88/88 ━━━━━━━━━━━━━━━━━━━━ 730s 8s/step - boundary_logits_loss: 0.0216 - loss: 1.2600 - sem_logits_loss: 1.2535
Epoch 12/20


=0.151  valBCE=135.1484  per-class IoU=[0.641 0.064 0.002 0.036 0.166 0.062 0.087]
88/88 ━━━━━━━━━━━━━━━━━━━━ 717s 8s/step - boundary_logits_loss: 0.0213 - loss: 1.2670 - sem_logits_loss: 1.2606
Epoch 15/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0204 - loss: 1.2511 - sem_logits_loss: 1.2450


Epoch 13/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0211 - loss: 1.2144 - sem_logits_loss: 1.2080[Eval] valPA=0.576  valmIoU=0.183  valBCE=129.4991  per-class IoU=[0.653 0.065 0.087 0.098 0.1   0.116 0.165]
88/88 ━━━━━━━━━━━━━━━━━━━━ 693s 8s/step - boundary_logits_loss: 0.0210 - loss: 1.2466 - sem_logits_loss: 1.2403
Epoch 14/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0208 - loss: 1.2638 - sem_logits_loss: 1.2576[Eval] valPA=0.617  valmIoU=0.151  valBCE=135.1484  per-class IoU=[0.641 0.064 0.002 0.036 0.166 0.062 0.087]
88/88 ━━━━━━━━━━━━━━━━━━━━ 717s 8s/step - boundary_logits_loss: 0.0213 - loss: 1.2670 - sem_logits_loss: 1.2606
Epoch 15/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0204 - loss: 1.2511 - sem_logits_loss: 1.2450
Epoch 13/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0211 - loss: 1.2144 - sem_logits_loss: 1.2080[Eval] valPA=0.576  valmIoU=0.183  valBCE=129.4991  per-class IoU=[0.653 0.065 0.087 0.098 0.1   0.116 0.165]
88/88 ━━━━━━━━━━━━━━━━━━━━ 693s 8s/step - boundary_logits_loss: 0.0210 - loss: 1.2466 - sem_logits_loss: 1.2403
Epoch 14/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0208 - loss: 1.2638 - sem_logits_loss: 1.2576[Eval] valPA=0.617  valmIoEpoch 13/20
Epoch 13/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0211 - loss: 1.2144 - sem_logits_loss: 1.2080[Eval] valPA=0.576  valmIoEpoch 13/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0211 - loss: 1.2144 - sem_logits_loss: 1.2080[Eval] valPA=0.576  valmIoU=0.183  valBCE=129.4991  per-class IoU=[0.653 0.065 0.087 0.098 0.1   0.116 0.165]
88/88 ━━━━━━━━━━━━━━━━━━━━ 693s 8s/step - boundary_logits_loss: 0.0210 - loss: 1.2466 - sem_logits_loss: 1.2403
Epoch 14/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0208 - loss: 1.2638 - sem_logits_loss: 1.2576[Eval] valPA=0.617  valmIoU=0.151  valBCE=135.1484  per-class IoU=[0.641 0.064 0.002 0.036 0.166 0.062 0.087]
88/88 ━━━━━━━━━━━━━━━━━━━━ 717s 8s/step - boundary_logits_loss: 0.0213 - loss: 1.2670 - sem_logits_loss: 1.2606
Epoch 15/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0204 - loss: 1.2511 - sem_logits_loss: 1.2450[Eval] valPA=0.595  valmIoU=0.157  valBCE=108.5102  per-class IoU=[0.642 0.011 0.015 0.135 0.169 0.11  0.019]
88/88 ━━━━━━━━━━━━━━━━━━━━ 692s 8s/step - boundary_logits_loss: 0.0209 - loss: 1.2545 - sem_logits_loss: 1.2482
Epoch 16/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0215 - loss: 1.2288 - sem_logits_loss: 1.22232025-10-31 21:37:37.527150: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.606  valmIoU=0.154  valBCE=117.8544  per-class IoU=[0.656 0.087 0.002 0.024 0.186 0.109 0.016]
88/88 ━━━━━━━━━━━━━━━━━━━━ 686s 8s/step - boundary_logits_loss: 0.0210 - loss: 1.2373 - sem_logits_loss: 1.2310
Epoch 17/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0207 - loss: 1.2452 - sem_logits_loss: 1.2390[Eval] valPA=0.615  valmIoU=0.201  valBCE=117.3357  per-class IoU=[0.66  0.015 0.025 0.124 0.169 0.171 0.241]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.201)
88/88 ━━━━━━━━━━━━━━━━━━━━ 706s 8s/step - boundary_logits_loss: 0.0208 - loss: 1.2490 - sem_logits_loss: 1.2427
Epoch 18/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0212 - loss: 1.2219 - sem_logits_loss: 1.2155[Eval] valPA=0.610  valmIoU=0.189  valBCE=107.8543  per-class IoU=[0.663 0.036 0.069 0.096 0.193 0.159 0.106]
88/88 ━━━━━━━━━━━━━━━━━━━━ 713s 8s/step - boundary_logits_loss: 0.0209 - loss: 1.2430 - sem_logits_loss: 1.2368
Epoch 19/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0214 - loss: 1.2364 - sem_logits_loss: 1.2300[Eval] valPA=0.622  valmIoU=0.183  valBCE=134.0219  per-class IoU=[0.664 0.01  0.007 0.064 0.186 0.145 0.203]
88/88 ━━━━━━━━━━━━━━━━━━━━ 714s 8s/step - boundary_logits_loss: 0.0207 - loss: 1.2342 - sem_logits_loss: 1.2280
Epoch 20/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0212 - loss: 1.2194 - sem_logits_loss: 1.2130[Eval] valPA=0.602  valmIoU=0.198  valBCE=123.2329  per-class IoU=[0.655 0.054 0.118 0.13  0.123 0.082 0.225]
88/88 ━━━━━━━━━━━━━━━━━━━━ 670s 8s/step - boundary_logits_loss: 0.0206 - loss: 1.2123 - sem_logits_loss: 1.2061