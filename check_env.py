import os
import cv2

from third_party.dwpose import DWposeDetector
from datasets import Processor


# sample_fns = os.listdir("./samples")
# dwprocessor = DWposeDetector()
#
# for fn in sample_fns:
#     if "cloth" not in fn:
#         continue
#     in_image = cv2.imread(f"./samples/{fn}")
#     in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
#     detected_map = dwprocessor(in_image)
#     detected_map = cv2.cvtColor(detected_map, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f"./tmp_dw_pose_{fn}", detected_map)

proc = Processor(
    root="/cfs/yuange/datasets/xss/trousers/",
    out_dir="/cfs/yuange/datasets/xss/standard/trousers/",
    extract_keys=["dwpose", ],
    is_debug=True,
)
proc.run()
