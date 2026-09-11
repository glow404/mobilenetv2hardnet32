import cv2
import numpy as np

from runtime import (
    build_sift,
    detect_sift_keypoints,
    patchable_keypoints,
)


class GTFeatureExtractor:

    def __init__(self, config):
        self.config = config
        self.sift = build_sift(config)

    def detect_keypoints(self, image):
        """
        image:
            numpy uint8
            H×W

        return
            list[cv2.KeyPoint]
        """

        keypoints = detect_sift_keypoints(
            image,
            self.sift,
            self.config
        )

        max_keypoints = int(
            self.config.get(
                "keypoint_filter",
                {}
            ).get(
                "max_keypoints",
                400
            )
        )

        if max_keypoints > 0 and len(keypoints) > max_keypoints:

            order = np.argsort(
                [
                    kp.response
                    for kp in keypoints
                ]
            )[::-1][:max_keypoints]

            keypoints = [
                keypoints[int(i)]
                for i in order
            ]

        return keypoints

    def extract_patches(
            self,
            image,
            keypoints,
    ):
        """
        image:
            numpy uint8
            H×W

        keypoints:
            list[cv2.KeyPoint]

        return

            selected_keypoints

            patches
                N×32×32

            selected_indices
        """

        patch_cfg = self.config.get(
            "patch",
            {}
        )

        selected_keypoints, patches, selected_indices = patchable_keypoints(

            image,

            keypoints,

            {
                "patch": {

                    "crop_size":
                        int(
                            patch_cfg.get(
                                "crop_size",
                                64
                            )
                        ),

                    "out_size":
                        int(
                            patch_cfg.get(
                                "out_size",
                                32
                            )
                        ),

                    "min_overlap_ratio":
                        float(
                            patch_cfg.get(
                                "min_overlap_ratio",
                                0.55
                            )
                        ),

                    "normalize":
                        bool(
                            patch_cfg.get(
                                "normalize",
                                True
                            )
                        ),
                }
            }
        )

        return (
            selected_keypoints,
            patches,
            selected_indices
        )