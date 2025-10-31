88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 9s/step - boundary_logits_loss: 0.3750 - loss: 2.0863 - sem_logits_loss: 1.7113d:\animal_data\img_segment\model_train.py:369: DeprecationWarning: 'mode' parameter is deprecated and will be removed in Pillow 13 (2026-10-15)
  mask = Image.fromarray(mask, mode="L").resize(
2025-10-31 10:53:31.247563: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.205  valmIoU=0.056  valBCE=0.2938  per-class IoU=[0.242 0.041 0.027 0.004 0.013 0.06  0.003]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.056)
88/88 ━━━━━━━━━━━━━━━━━━━━ 865s 10s/step - boundary_logits_loss: 0.3734 - loss: 2.0824 - sem_logits_loss: 1.7090
Epoch 2/3
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 9s/step - boundary_logits_loss: 0.0848 - loss: 1.4929 - sem_logits_loss: 1.40812025-10-31 11:08:13.143121: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[Eval] valPA=0.313  valmIoU=0.090  valBCE=0.1032  per-class IoU=[0.393 0.022 0.086 0.055 0.014 0.057 0.001]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.090)
88/88 ━━━━━━━━━━━━━━━━━━━━ 882s 10s/step - boundary_logits_loss: 0.0847 - loss: 1.4924 - sem_logits_loss: 1.4078
Epoch 3/3
88/88 ━━━━━━━━━━━━━━━━━━━━ 0s 9s/step - boundary_logits_loss: 0.0528 - loss: 1.3881 - sem_logits_loss: 1.3353[Eval] valPA=0.557  valmIoU=0.133  valBCE=0.0506  per-class IoU=[0.612 0.004 0.133 0.017 0.035 0.119 0.009]
Saved best to D:\animal_data\models\unet_boundary_best.keras (mIoU 0.133)

