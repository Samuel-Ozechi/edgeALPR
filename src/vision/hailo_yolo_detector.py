import time
import cv2
import numpy as np

from hailo_platform import (
    HEF,
    Device,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


class HailoYoloDetector:
    """
    Hailo-accelerated YOLO detector wrapper.

    Expected responsibility:
    - load HEF
    - preprocess image
    - run inference on Hailo
    - decode YOLO output
    - return same format as existing detectors
    """

    def __init__(
        self,
        hef_path: str,
        class_names: list[str],
        conf_threshold: float,
        model_type: str,
        input_size: tuple[int, int] = (640, 640),
        vdevice: VDevice = None,
    ):
        self.hef_path = hef_path
        
        # FIX: Ensure class_names is safely indexed as a list, even if passed as a set
        self.class_names = list(class_names) if not isinstance(class_names, list) else class_names
        
        self.conf_threshold = conf_threshold
        self.model_type = model_type
        self.input_size = input_size

        self.hef = HEF(hef_path)
        self.target = vdevice if vdevice is not None else VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(
            self.hef,
            configure_params
        )[0]
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32
        )
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        print(f"[INFO] Loaded Hailo model: {hef_path}")
        print(f"[INFO] Input shape: {self.input_info.shape}")

    def preprocess(self, image: np.ndarray):
        original_h, original_w = image.shape[:2]

        resized = cv2.resize(image, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.uint8)

        # Explicitly passing height and width in a structured key/value pair
        return tensor, {"width": original_w, "height": original_h}

    def infer(self, input_tensor):
        """
        Run Hailo inference.
        """
        input_data = {
            self.input_info.name: np.expand_dims(input_tensor, axis=0)
        }

        with self.network_group.activate():
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params
            ) as infer_pipeline:
                results = infer_pipeline.infer(input_data)

        return results

    def postprocess(self, raw_outputs, original_shape):
        """
        Decodes the Hailo NMS list output format.
        
        Expected raw_outputs structure: 
        A dictionary containing a list of length 1, containing an array sequence 
        where each index represents a class ID.
        Each row in the class array contains: [ymin, xmin, ymax, xmax, confidence]
        """
        detections = []
        
        # FIX: Robustly pull explicit dimensions to prevent axis-flipping bugs
        orig_w = original_shape["width"]
        orig_h = original_shape["height"]

        # Extract the array list from the Hailo output structure
        if isinstance(raw_outputs, dict):
            if not raw_outputs:
                return detections
            # Dynamically fetch the postprocess dictionary key (e.g., 'yolov8n/yolov8_nms_postprocess')
            key = list(raw_outputs.keys())[0]
            class_arrays = raw_outputs[key][0]
        elif isinstance(raw_outputs, (list, tuple)):
            if not raw_outputs:
                return detections
            class_arrays = raw_outputs[0]
        else:
            return detections

        # Iterate through each class array in the sequence
        for class_id, class_array in enumerate(class_arrays):
            if class_array.size == 0:
                continue

            for row in class_array:
                if row.size < 5:
                    continue
                ymin, xmin, ymax, xmax, confidence = map(float, row[:5])

                # Filter out low-confidence hits
                if confidence < self.conf_threshold:
                    continue

                # Clip normalized coordinates securely to [0.0, 1.0] range
                ymin = max(0.0, min(1.0, ymin))
                xmin = max(0.0, min(1.0, xmin))
                ymax = max(0.0, min(1.0, ymax))
                xmax = max(0.0, min(1.0, xmax))

                # Denormalize coordinates directly to original image dimensions
                x1 = int(round(xmin * orig_w))
                y1 = int(round(ymin * orig_h))
                x2 = int(round(xmax * orig_w))
                y2 = int(round(ymax * orig_h))

                # Ensure valid box dimensions
                if x2 <= x1 or y2 <= y1:
                    continue

                detection = {
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                }

                # Map class details for the multi-class vehicle detector
                if self.model_type == "vehicle":
                    detection["class_id"] = class_id
                    if class_id < len(self.class_names):
                        detection["class_name"] = self.class_names[class_id]
                    else:
                        detection["class_name"] = "unknown"
                else:
                    # Default class parameters for single class tracking (like plates)
                    detection["class_id"] = 0
                    detection["class_name"] = self.class_names[0] if self.class_names else "plate"

                detections.append(detection)

        # Sort results highest confidence first to match Ultralytics detector style
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)

    def detect(self, image: np.ndarray):
        start = time.perf_counter()

        input_tensor, original_shape = self.preprocess(image)
        raw_outputs = self.infer(input_tensor)
        
        # Concise debug logger
        try:
            if isinstance(raw_outputs, dict):
                for k, v in raw_outputs.items():
                    if isinstance(v, list) and v:
                        print(f"[DEBUG HAILO OUTPUT] Processing key={k} | Class arrays counts={len(v[0])}")
        except Exception as e:
            print("[DEBUG HAILO OUTPUT] Logging context skipped:", e)
            
        detections = self.postprocess(raw_outputs, original_shape)
        latency_ms = (time.perf_counter() - start) * 1000

        return detections, latency_ms

    def select_best_vehicle(self, detections):
        if not detections:
            return None
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]

    def select_best_plate(self, detections):
        if not detections:
            return None
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]