# Object detection demo

This is a demo for an HMGCC Co-Creation bid. The challenge is set out in the [challenge document](https://co-creation.hmgcc.gov.uk/api/v1/files/challenges/2244b79e-62ac-4b9a-a7e5-dc99928ef4ac/56f39b24_13784797_1789984142.pdf), which asks: "Can you develop a machine learning-enabled solution that helps engineers identify, characterise, and understand electronic components using only x-ray images?"

Our bid is an X-ray analysis workflow. This project covers the object detection stage of that workflow, locating the components in an image and assigning each one a class. Every prediction comes with a confidence score, so an operator can see how sure the model is about each box and which ones need a second look. The work was severely time limited, and the challenge X-rays are only provided to the winning provider, so this is a demo of the method, not a detector trained or scored on the challenge X-rays. Later stages, such as tracing wires and tracks, building a circuit diagram, and operator review, are outside this demo.

This demo was put together in a very short window, but the full project runs over 12 weeks planned in sprints, each with a set outcome, so the stages above are scheduled in rather than left out, and the general quality and organisation of the work will be much stronger.

## Dataset

We trained and evaluated on [CGHD](https://huggingface.co/datasets/lowercaseonly/cghd) (Circuit Graph Hand-Drawn). These are photographs of hand-drawn circuit diagrams, with a bounding box and a class for each symbol.

CGHD mirrors the detection stage in shape. Both tasks ask a model to find many small circuit components in one image and name them. CGHD also has a long list of classes with a severe imbalance in how often each one appears, which is the same kind of pressure we expect the X-ray stage to face.

## What the demo shows

The two notebooks are one pass through the workflow we would use to build that stage.

### 1. Exploratory analysis

`ObjectClassification/eda.ipynb` Exploratory Data Analysis records the evidence we used to choose a model. Given our focus is the components and their connections, not the writing on the page, we left the text labels out. Removing them drops about `36.66%` of the boxes and still leaves over `150,000` for modelling.

What we found:

- There are `58` component classes once text is removed, and the class imbalance is severe, with some classes making up about `0.01%` of instances.
- There's a huge amount of diversity in image size, so images need resizing to one consistent input size.
- A typical image holds many objects, a median of `32` and a mean of `48` per image, with some images far more crowded.
- Boxes are small relative to the image. Height and width are mostly `2–5%` of the image (median `3%`), and a typical box covers about `0.09%` of the image area.
- Overlap is limited. About half the images have some overlap, and in the worst cases about `0.6` of that image's boxes overlap.
- Most images have EXIF orientation `1`, but some files are stored rotated while the boxes were drawn on the upright photo, so every image is opened with `exif_transpose` so the pixels line up with the boxes.
- Only two boxes sit outside their image.

### 2. Model choice

The model choice follows from that evidence. `ObjectClassification/detr.ipynb` fine tunes RT-DETR (`PekingU/rtdetr_r50vd`, a ResNet-50 backbone), a transformer based, anchor-free detector.

- Our images hold a varying number of objects, and RT-DETR's object queries let it emit a varying number of boxes.
- It replaces non-maximum suppression (NMS) with bipartite matching, so closely packed boxes are not discarded prematurely.
- A fixed number of object queries avoids anchor flooding, giving under-represented classes a fairer chance across all 58 classes.
- Anchor-free boxes can be any size and aspect ratio, which suits our small, distinct objects.

In the full bid, EDA, would be used the same way to shortlist more than one candidate model. Each candidate would then be trained and scored on the same fixed metrics, giving a fair comparison of how each model performs on the dataset.

The training setup for this run:

- An iterative (multilabel) stratified split of 70/10/20 for train, validation and test (`2,285`, `330` and `678` images), so each split has the same share of every class. Sklearn's `train_test_split` can't stratify here because each image holds many classes.
- Images resized to `640x640` with ImageNet normalisation, and boxes converted to normalised centre coordinates (`center_x, center_y, width, height`).
- AdamW with a learning rate of `1e-5` for the pretrained layers and `1e-4` for the reinitialised class-score layers (`class_embed`, `enc_score_head`, `denoising_class_embed`), weight decay of `1e-4`, batch size `64`, and gradient clipping at `0.1`.
- Up to `40` epochs with early stopping on the validation cost (patience of `8`). This run used all 40, and the weights with the lowest validation cost were kept.

### 3. A fixed evaluation

Every candidate is judged on the same list:

- per-class average precision at an overlap (IoU) of 0.5
- mean average precision at 0.5
- mean average precision from 0.5 to 0.95
- a precision-recall curve as confidence falls
- sample images with the predicted class and confidence drawn on each box

Each box's confidence is the highest sigmoid class score for that object query, and that same score picks the predicted class, so every detection comes with both a class and a confidence.

On this dataset, one training run reached mAP@0.5 of `0.433` and mAP@0.5:0.95 of `0.312`.

- `12` classes have an average precision above `0.7`, with `resistor`, `transistor.bjt` and `operational_amplifier` near `0.9`. Nine classes have an average precision of `0`, so those symbols are not being detected.
- There is a strong positive correlation between how many boxes a class has in the test set and its average precision (R = `0.752`). Classes with only a few dozen boxes sit near 0, while classes with around a thousand or more mostly sit around `0.85` to `0.9`. Rare classes count the same as common ones in mAP, which is what pulls the headline number down, and more samples for those classes is very likely to improve them.
- The drop from `0.433` to `0.312` tells us the model finds the right symbol more often than it places the box tightly.
- The precision-recall curve stays near a precision of 1 until a recall of about `0.6`, and is still around `0.9` at about `0.7`. So when the model is confident it is usually right. It ends near a recall of `0.8` with a precision of about `0.15`, so the least confident predictions are mostly wrong and about a fifth of the ground-truth boxes are never found.

The full project would keep that list and likely add metrics this demo does not report (due to time constraints):

- mean average precision weighted by how many boxes a class has, reported beside the unweighted figure, so a rare class cannot move the headline number on its own
- average precision split by box size, because our objects are small relative to the image
- precision, recall, and F1 at the confidence threshold actually used to draw a box, overall and per class
- for each class, the share of ground-truth boxes missed, kept separate from the share given the wrong class
- mean overlap of the boxes that were the right class, so a loose box is not hidden inside mAP@0.5:0.95
- a check that the confidence score matches how often the box is right (calibration), since low-confidence detections are what a reviewer would be asked to look at again

The notebooks open CGHD images with Pillow's `Image.open`, and those files are JPEG and PNG. Pillow can also open TIFF, which is the other format named in the challenge, but no TIFF X-ray was included in this run.

## Not in this demo

These stages were left out because of time. They will be part of a full implementation beyond this demo:

- Hyperparameter tuning, rather than the single training setup used here. That means more than one learning rate, weight decay, and epoch budget, with the same split and the same metrics each time.
- Training several detectors, not only RT-DETR, on that same split and scoring every one on the fixed metrics above.
- An augmentation study: flips, small rotations, and brightness changes, with the boxes transformed to match, then keeping only the augmentations that improve the fixed metrics on the validation set.
- A larger validation set than the 10% split used here, and a separate pass that sets the ideal confidence threshold from the validation predictions. In this demo that cutoff is `0.5`, chosen before we looked at the results, and it only decides which boxes are drawn on the sample images at inference. It does not change the scores above, because average precision walks every prediction of a class from the most confident to the least, and mean average precision averages those walks across every class.
- Remedies for the class imbalance the evaluation already measures. Class weights (scale the loss by the inverse frequency of each class, so it penalizes mistakes on rare classes more) would scale the classification part of the loss, so a mistake on a rare class costs more than a mistake on a common one such as `junction`. Upsampling would show training images that contain rare classes more often than they occur in the data. Either change would be judged by whether the average precision of those rare classes rises on the same test split.
- The later workflow stages: wire and track segmentation using the CGHD segmentation maps, circuit-diagram construction, and human review of low-confidence boxes.

## Layout

| Path | Role |
| --- | --- |
| `ObjectClassification/eda.ipynb` | Dataset checks that decide which models are worth training |
| `ObjectClassification/detr.ipynb` | One detector trained and scored on the fixed metrics |
| `data/` | The CGHD download and exported samples, kept out of git |
| `rtdetr_cghd/` | Local fine-tuned weights, kept out of git because of file size |

Both notebooks read the dataset location from `DATA_PATH` in a local `.env` file. Training was run on a Colab GPU.

The system does not decide whether a device is safe, and it does not recommend an action. Human operators remain responsible for every decision. This demo only shows that an object detection stage can be trained and scored with a confidence attached to each box. It does not show that stage working on X-ray images.
