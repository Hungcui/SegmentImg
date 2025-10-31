Classes (7): ['background', 'cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
2025-10-31 14:01:38.507789: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/3
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 9s/step - boundary_logits_loss: 0.3704 - loss: 2.1308 - sem_logits_loss: 1.76042025-10-31 14:16:47.109342: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.192  valmIoU=0.052  valBCE=1.4060  per-class IoU=[0.217 0.047 0.04  0.057 0.003 0.003 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.052)
88/88 ━━━━━━━━━━━━━━━━━━━━ 910s 10s/step - boundary_logits_loss: 0.2269 - loss: 1.7961 - sem_logits_loss: 1.5693
Epoch 2/3
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 9s/step - boundary_logits_loss: 0.0841 - loss: 1.5070 - sem_logits_loss: 1.42292025-10-31 14:31:31.794200: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.520  valmIoU=0.133  valBCE=0.0909  per-class IoU=[0.568 0.009 0.14  0.007 0.101 0.105 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.133)
88/88 ━━━━━━━━━━━━━━━━━━━━ 885s 10s/step - boundary_logits_loss: 0.0728 - loss: 1.4667 - sem_logits_loss: 1.3939
Epoch 3/3
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 9s/step - boundary_logits_loss: 0.0499 - loss: 1.3877 - sem_logits_loss: 1.3378[Eval] valPA=0.403  valmIoU=0.108  valBCE=0.0922  per-class IoU=[0.463 0.021 0.137 0.03  0.025 0.083 0.   ]
88/88 ━━━━━━━━━━━━━━━━━━━━ 862s 10s/step - boundary_logits_loss: 0.0486 - loss: 1.4164 - sem_logits_loss: 1.3678