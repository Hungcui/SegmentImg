Classes (7): ['background', 'cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
2025-10-31 17:33:20.832462: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 6s/step - boundary_logits_loss: 0.1062 - loss: 1.8652 - sem_logits_loss: 1.75892025-10-31 17:44:20.172324: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.613  valmIoU=0.092  valBCE=32.8547  per-class IoU=[0.615 0.    0.    0.    0.032 0.    0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.092)
88/88 ━━━━━━━━━━━━━━━━━━━━ 660s 7s/step - boundary_logits_loss: 0.0706 - loss: 1.6599 - sem_logits_loss: 1.5892
Epoch 2/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 6s/step - boundary_logits_loss: 0.0495 - loss: 1.5209 - sem_logits_loss: 1.47132025-10-31 17:55:28.463592: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.617  valmIoU=0.105  valBCE=25.8505  per-class IoU=[0.623 0.    0.    0.002 0.108 0.    0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.105)
88/88 ━━━━━━━━━━━━━━━━━━━━ 668s 8s/step - boundary_logits_loss: 0.0420 - loss: 1.4547 - sem_logits_loss: 1.4128
Epoch 3/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 6s/step - boundary_logits_loss: 0.0364 - loss: 1.4281 - sem_logits_loss: 1.3917[Eval] valPA=0.618  valmIoU=0.111  valBCE=20.4404  per-class IoU=[0.624 0.    0.027 0.    0.085 0.039 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.111)
88/88 ━━━━━━━━━━━━━━━━━━━━ 660s 7s/step - boundary_logits_loss: 0.0353 - loss: 1.4226 - sem_logits_loss: 1.3874
Epoch 4/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0335 - loss: 1.4014 - sem_logits_loss: 1.36782025-10-31 18:17:41.374571: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.599  valmIoU=0.125  valBCE=20.3359  per-class IoU=[0.631 0.    0.    0.    0.141 0.101 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.125)
88/88 ━━━━━━━━━━━━━━━━━━━━ 673s 8s/step - boundary_logits_loss: 0.0341 - loss: 1.4118 - sem_logits_loss: 1.3777
Epoch 5/20
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step - boundary_logits_loss: 0.0327 - loss: 1.3792 - sem_logits_loss: 1.3465[Eval] valPA=0.609  valmIoU=0.126  valBCE=19.9342  per-class IoU=[0.631 0.    0.018 0.    0.135 0.098 0.   ]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.126)


Pay attention to line 390