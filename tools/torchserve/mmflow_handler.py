# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import cv2
import mmcv
import torch
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from ts.torch_handler.base_handler import BaseHandler

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow


class MMFlowHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_model(self.config_file, checkpoint, self.device)
        self.model = revert_sync_batchnorm(self.model)
        self.initialized = True

    def preprocess(self, data):
        videos = []
        for row in data:
            content = row.get('data') or row.get('body')
            if isinstance(content, str):
                content = base64.b64decode(content)
            videos.append(content)
        return videos

    def inference(self, data, *args, **kwargs):
        results = []
        for i, content in enumerate(data):
            src_file = os.getcwd() + 'src' + str(i) + '.mp4'
            res_file = os.getcwd() + 'res ' + str(i) + '.mp4'
            with open(src_file, 'wb') as file:
                file.write(content)
            cap = cv2.VideoCapture(src_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(res_file, fourcc, fps, size)

            frames = []
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                frames.append(frame)

            for i in range(len(frames) - 1):
                img1 = frames[i]
                img2 = frames[i + 1]
                # estimate flow
                result = inference_model(self.model, img1, img2)
                flow_map = visualize_flow(result, None)
                videoWriter.write(mmcv.rgb2bgr(flow_map))

            os.remove(src_file)
            results.append(res_file)
        return results

    def postprocess(self, data):
        output = []

        for res_path in data:
            with open(res_path, 'rb') as file:
                content = file.read()
            output.append(content)
            os.remove(res_path)
        return output
