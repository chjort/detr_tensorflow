import itertools

import tensorflow as tf

from chambers.utils.boxes import get
from chambers.utils.tf import round

boxes = tf.convert_to_tensor([[236.98, 142.51, 261.68, 212.01],
                              [7.03, 167.76, 156.35, 262.63],
                              [557.21, 209.19, 638.56, 287.92],
                              [358.98, 218.05, 414.98, 320.88],
                              [290.69, 218., 352.52002, 316.48],
                              [413.2, 223.01, 443.37003, 304.37],
                              [317.4, 219.24, 338.97998, 230.83],
                              [412.8, 157.61, 465.84998, 295.62],
                              [384.43, 172.21, 399.55, 207.95001],
                              [512.22, 205.75, 526.95996, 221.72],
                              [493.1, 174.34, 513.39, 282.65],
                              [604.77, 305.89, 619.11005, 351.6],
                              [613.24, 308.24, 626.12, 354.68],
                              [447.77, 121.12, 461.74, 143.],
                              [549.06, 309.43, 585.74, 399.09998],
                              [350.76, 208.84, 362.13, 231.39],
                              [412.25, 219.02, 421.88, 231.54001],
                              [241.24, 194.99, 255.46, 212.62001],
                              [336.79, 199.5, 346.52002, 216.23],
                              [321.21, 231.22, 446.77, 320.15]], dtype=tf.float32)

formats = ("xyxy", "yxyx", "cxcywh", "xywh")
combinations = list(itertools.combinations(formats, 2))

f0 = "xyxy"
f1, f2 = combinations[2]
for f1, f2 in combinations:
    if f1 != f0:
        f0_to_f1 = get("box_{}_to_{}".format(f0, f1))
        boxes_f1 = f0_to_f1(boxes)
    else:
        boxes_f1 = boxes

    f1_to_f2 = get("box_{}_to_{}".format(f1, f2))
    f2_to_f1 = get("box_{}_to_{}".format(f2, f1))

    boxes_f1 = round(boxes_f1, 3)
    boxes_f1_ = round(f2_to_f1(f1_to_f2(boxes_f1)), 3)
    tf.assert_equal(boxes_f1, boxes_f1_)
