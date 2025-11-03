*tớ đã up present_plan.md trên github rồi nhé, mn xem có j cần chỉnh sửa ko và nhớ xem có j bất ổn ko

Title: U-net training for Semantic Segmentation

Purpose for this project ? (form report?)

Data: 
    Preparation: 
        6 type animal data - each animal data have 4 seperate file
            ImageSets: train and val txt include set of train and val image's name
            JPEGImage: original image
            SegmentationClass: segmented image
            SegmentationObject: each animal segmentation seperated with different type (should be shown in here?????)
        labelmap.txt

    Preprocessing:
        Clean-up data: 
            Check each image has as a group (original, segmentation)
            Check image size -> resize to 512 x 512 if not 512 x 512 img
        limited data(why limited) 60 img train / 15 img val / 15 img test(per animal) as for our model -> data augmentation for improving diversity
        

Diagram pipeline:
Data -> Augment/Normalize -> Model (Attention U-net, Standard U-net, Backbone U-net) -> Output: Semantic and boundary logit -> Losses -> eval callback(mIoU) -> Inference (TTA) -> Post-processing

    Boundary logit for ?
        predicts a per-pixel boundary probability map
        enable reliable instance splitting


Each unet model
    Attention U-net: skip connection for supress not relevant region -> staying efficient and improve model
    Backbone U-net: pretrained encoders; uses compound scaling for better accuracy-efficiency trade-off
    Standard U-net:

Accuracy/Losses (from report)
    Compare each u-net model

What have done?
    Developed segmentation training system
    Trained multiple u-net model (with "test" image from model)
    Turn semantic + boundary maps into per-animal instances.
        Use distance-transform peaks as markers and watershed to split touching animals — a textbook instance-from-semantic approach.

What can be improved?
    Increase number of epoches
    Increase dataset quantity
    Update model form semantic segmenation -> instance segmentation
    Optimize model: quantize/pruning