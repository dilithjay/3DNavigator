import numpy as np
import time

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
import time
import pyautogui


labelmap_path = "labelmap.txt"
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

tf.keras.backend.clear_session()
model = tf.saved_model.load('inference_graph/saved_model')

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


vc = cv2.VideoCapture(0)

# Set video capture resolution
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

size = pyautogui.size()

counts = (0, 0, 0, 0)       # the number of fingers detected in the last 4 frames. Used for checking consistency
num = 0                     # most common count out of counts (mode of the list)
prev_col = 2                # index of color out of 1 or 2. blue: 1, read: 2
prev_loc = (960, 540)       # used to keep track of previous location of head. Starting at the center of the screen

def run_inference(model, cap):
    global num
    global prev_col
    global prev_loc
    
    t = time.time()
    ret, image_np = cap.read()
    img = image_np
    
    # Actual detection.
    output_dict = run_inference_for_single_image(model, img)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=.2,
        line_thickness=5)
    
    print('inference time:', time.time() - t, "seconds")
    
    detection_scores = output_dict['detection_scores'][:]
    maxes = []
    max_boxes = []
    col = output_dict['detection_classes'][np.argmax(detection_scores)]
    
    # Get list of valid detections
    while len(maxes) < 3:
        index = np.argmax(detection_scores)
        # check if accuracy is acceptable
        if detection_scores[index] > 0.2:
            # box format: ymin, xmin, ymax, xmax
            box = output_dict['detection_boxes'][index]
            y_avg = (box[0] + box[2])/2
            x_avg = (box[1] + box[3])/2
            detection_scores[index] = -1
            # check if distance to previously detected boxes is not too small (to avoid multiple detections)
            if all((i[0] - y_avg)**2 + (i[1] - x_avg)**2 > 0.006 for i in maxes):
                maxes.append([y_avg, x_avg])
            else:
                print("overlap")
        else:
            break
    
    # Update the list of counts of detections
    counts.pop(0)
    counts.append(len(maxes))
    
    # num is used to prevent the effects of bad detections
    # It's equal to the most common count in the last 4 frames
    num = max(set(counts), key=counts.count)
    
    # If there are detections,
    if len(maxes) > 0:
        # first finger is the leftmost detected finger
        first_finger = sorted(maxes, key=lambda x: x[1])[0]
        loc = (first_finger[1]*1920, first_finger[0]*1080)
        
        # one finger detected,
        if num == 1 and len(maxes) == 1:
            # t = 0.1 * ((loc[0] - prev_loc[0])**2 + (loc[1] - prev_loc[1])**2)/4852800
            pyautogui.moveTo(1920 - loc[0], loc[1], _pause=False)
            prev_loc = loc
            if prev_col == 2 and col == 1:
                pyautogui.mouseDown(button='left')
            elif prev_col == 1 and col == 2:
                pyautogui.mouseUp(button='left')
        
        
        # two fingers detected
        elif num == 2 and len(maxes) == 2:
            displacement = int(prev_loc[1] - loc[1])
            # print(displacement)
            pyautogui.scroll(int(displacement//2), _pause=False)
            
            
        # three fingers detected
        elif num == 3 and len(maxes) == 3:
            pyautogui.mouseDown(button='middle')
            pyautogui.moveTo(1920 - loc[0], loc[1], _pause=False)
            prev_loc = loc
            if prev_col == 2 and col == 1:
                pyautogui.mouseUp(button='middle')
                pyautogui.keyDown('ctrl')
                pyautogui.keyDown('alt')
                pyautogui.mouseDown(button='left')
            elif prev_col == 1 and col == 2:
                pyautogui.mouseUp(button='left')
                pyautogui.keyUp('alt')
                pyautogui.keyUp('ctrl')
                pyautogui.mouseDown(button='middle')
        else:
            pyautogui.mouseUp(button='middle')
            pyautogui.mouseUp(button='left')
            pyautogui.keyUp('alt')
            pyautogui.keyUp('ctrl')
            
        prev_col = col
        print(["blue", "red"][col - 1], ":", num)
    
    return img

while vc.isOpened():
    t1 = time.time()
    img = run_inference(model, vc)
    """
    cv2.imshow('image', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        vc.release()
        cv2.destroyAllWindows()
        break"""
    # print("FPS:", 1/(time.time() - t1))
