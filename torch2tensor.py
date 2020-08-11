# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import struct

import torch
from lenet_pytorch import LeNet


def main():
    model = LeNet().to("cuda:0")
    model.load_state_dict(torch.load("/opt/tensorrt_models/torch/lenet/lenet.pth"))
    model.eval()
    print("Load the specified pre-training weight successfully.")

    f = open("/opt/tensorrt_models/torch/lenet/lenet.wts", "w")
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    
    print("The weight conversion has been completed and saved to `/opt/tensorrt_models/torch/lenet/lenet.wts`.")



if __name__ == "__main__":
    main()
