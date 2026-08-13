"""
Harvard-Fairseg Dataset Creator
"""


import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from dataset_creator import DatasetCreator
import random
from PIL import Image, ImageDraw, ImageFont
import argparse

class HarvardFairsegRegnCreator(DatasetCreator):
    def __init__(self, npz_dir, save_dir, font_path):
        self.npz_dir = npz_dir
        self.save_dir = save_dir
        self.targets = {"optic disc": -1, "optic cup": -2}
        self.colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "brown"]
        self.color_list = {"optic disc":"", "optic cup":""}
        self.base_font_size = 20
        self.base_line_width = 2
        self.font_path = font_path

    def create_bboxes(self, mask):
        bbox_info = {}
        for target, value in self.targets.items():
            locations = np.where(mask == value)
            if locations[0].size > 0:
                # Bounding box: [x_min, y_min, x_max, y_max]
                bbox = [np.min(locations[1]), np.min(locations[0]), np.max(locations[1]), np.max(locations[0])]
                normalized_bbox = self.normalize_coordinates(mask.shape, bbox)
                bbox_info[target] = normalized_bbox
        return bbox_info

    def create_segmentations(self, mask, save_path, img_width, img_height):
        # Mapping target values to names
        region_mapping = {
            -1: "optic disc",  # Optic disc has value -1
            -2: "optic cup"   # Optic cup has value -2
        }

        segmentation_paths = []
        counter = 1
        os.makedirs(save_path, exist_ok=True)

        for value, region_type in region_mapping.items():
            mask_value = (mask == value).astype(np.uint8)

            # Create segmentation image
            segmentation_img = Image.fromarray((mask_value * 255).astype(np.uint8))
            segmentation_img = segmentation_img.resize((img_width, img_height), Image.NEAREST)

            # Save segmentation image
            seg_filename = f"segm_{counter}_{region_type.replace(' ', '_')}.png"
            seg_filepath = os.path.join(save_path, seg_filename)
            segmentation_img.save(seg_filepath)

            # Add segmentation path to the list
            segmentation_paths.append({
                "ID": f"segm_{counter}",
                "region_type": region_type,
                "segmentation_path": os.path.join("segmentations", os.path.relpath(seg_filepath, save_path))
            })
            counter += 1
        random.shuffle(segmentation_paths)
        return segmentation_paths


    def create_annotations(self, mask, save_path, img_width, img_height):
        # Create bounding boxes and segmentations
        bboxes = self.create_bboxes(mask)
        segmentations = self.create_segmentations(mask, save_path, img_width, img_height)

        return {
            "bounding_boxes": bboxes,
            "segmentations": segmentations
        }

    def create_metadata(self, file_name, bbox_info, segmentations):
        annotations = {
            "bounding_boxes": [
                {
                    "bounding_box": {
                        "top_left": [bbox[0], bbox[1]],
                        "bottom_right": [bbox[2], bbox[3]]
                    },
                    "region_type": target,
                    "annotation_ID": i,  
                    "color": self.color_list.get(target, 'black') ,
                }
                for i, (target, bbox) in enumerate(bbox_info.items(), start=1)
            ],
            "segmentations": segmentations
        }

        return {
            "image_path": f"{file_name}/visualization_{file_name}.png",
            "image_type": "Fundus Image",
            "annotations": annotations,  # Bounding boxes stored here as a list
            "metadata": {
                "dataset_id": file_name,
                "additional_info": ""
            }
        }


    def create_json_data(self, file_name, bbox_info, segmentations, annotations=None, metadata=None):
        if metadata is None:
            metadata = self.create_metadata(file_name, bbox_info, segmentations)
        # print(metadata)
        metadata["annotations"] = metadata["annotations"] 
        # print(metadata["annotations"])
        return metadata


    def shuffle_bboxes(self, bbox_info):
        annotations_list = [(target, bbox) for target, bbox in bbox_info.items()]
        random.shuffle(annotations_list)
        shuffled_bbox_info = {target: bbox for target, bbox in annotations_list}
    
        return shuffled_bbox_info
    

    def create_instance(self, npz_path):
        file_name = os.path.splitext(os.path.basename(npz_path))[0]
        directory_name = f"{self.save_dir}/{file_name}"
        os.makedirs(directory_name, exist_ok=True)
        os.makedirs(f"{directory_name}/annotated", exist_ok=True)

        data = np.load(npz_path)
        image = data['slo_fundus']
        mask = data['disc_cup_mask']
        bbox_info = self.create_bboxes(mask)

        shuffled_bbox_info = self.shuffle_bboxes(bbox_info)

        # Save visualization and bounding box overlay
        vis_filename = f"{directory_name}/visualization.png"
        bbox_filename = f"{directory_name}/annotated/annotated_bounding_box.png"
        self.save_visualization(image, vis_filename)
        self.save_bbox_overlay(image, shuffled_bbox_info, bbox_filename)

        # Create segmentations and save them
        segmentation_dir = f"{directory_name}/segmentations"
        segmentations = self.create_segmentations(mask, segmentation_dir, image.shape[1], image.shape[0])

        # Save metadata JSON
        json_filename = f"{directory_name}/information.json"
        json_data = self.create_json_data(file_name, shuffled_bbox_info, segmentations)
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=4)

    def process_all_npz_files(self):
        for file in os.listdir(self.npz_dir):
            if file.endswith(".npz"):
                self.create_instance(os.path.join(self.npz_dir, file))
        print("Processing complete.")

    def normalize_coordinates(self, shape, bbox):
        height, width = shape
        return [
            bbox[0] / width, bbox[1] / height,
            bbox[2] / width, bbox[3] / height
        ]

    def save_visualization(self, image, filename):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_bbox_overlay(self, image, bbox_info, filename):
        img_pil = Image.fromarray(np.uint8(image))
        if img_pil.mode != 'RGBA':
            img_pil = img_pil.convert('RGBA')
        
        # Load the font and line width
        font_size = max(self.base_font_size, int(self.base_font_size * min(img_pil.size) / 1000))
        line_width = max(self.base_line_width, int(self.base_line_width * min(img_pil.size) / 400))
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Create the bounding box overlay with annotations
        self.create_visualization_image(img_pil, bbox_info, font, line_width)
        
        # Save the final image
        img_pil.save(filename)
    
    def create_visualization_image(self, img_pil, annotations, font, line_width):
        draw = ImageDraw.Draw(img_pil, "RGBA")
        
        # Loop through bounding boxes and draw them on the image
        for i, (bbox_type, bbox) in enumerate(annotations.items(), start=1):
            # Normalize coordinates
            x1, y1, x2, y2 = bbox
            
            # Convert normalized coordinates to pixel values (multiply by image width/height)
            width, height = img_pil.size
            x1, y1 = int(x1 * width), int(y1 * height)
            x2, y2 = int(x2 * width), int(y2 * height)

            # Assign a random color for each bounding box
            color = random.choice(self.colors)
            self.color_list[bbox_type] = color
            
            # Draw the rectangle for the bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # Draw the label (index) next to the bounding box
            draw.text((x1 + 10, y1 + 10), str(i), fill=color, font=font)


        # If you want a legend with indices on the image, you can add this here too.
        legend_elements = [Line2D([0], [0], color=color, lw=4, label=str(i)) for i, color in enumerate(self.colors, start=1)]
        plt.legend(handles=legend_elements, loc='upper right')


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Create Dataset for Harvard Fairseg Regn")
    
    parser.add_argument('--base_dir', required=True, help="Base directory for NPZ files")
    parser.add_argument('--save_dir', required=True, help="Directory to save the processed data")
    parser.add_argument('--font_path', type=str, required=True, help="Font path to use for text rendering")
    
    args = parser.parse_args()
    npz_dir = f"{args.base_dir}/Test" 

    creator = HarvardFairsegRegnCreator(
        npz_dir=npz_dir,
        save_dir=args.save_dir,
        font_path=args.font_path
    )

    creator.process_all_npz_files()



# python dataset_creator_harvard_fairseg.py --base_dir "/Users/tinad/Desktop/data/harvard" --save_dir "/Users/tinad/Desktop/data/Goji/harvard" --font_path "/Users/tinad/Downloads/SmileySans-Oblique.ttf"