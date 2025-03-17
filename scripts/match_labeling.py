import cv2
import numpy as np
import json
import os
from pdf2image import convert_from_path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
cv2.setRNGSeed(12345)

def normalize_image(image, target_width=1000):
    h, w = image.shape[:2]
    scale = target_width / float(w)
    return cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def apply_clahe_preprocessing(image):
    """
    Convert to grayscale and apply CLAHE to enhance contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Convert back to BGR so feature matching can remain consistent
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def load_new_form_images(new_form_path, poppler_path=None, enhance=False):
    ext = os.path.splitext(new_form_path)[1].lower()
    images = []
    if ext == ".pdf":
        pages = convert_from_path(new_form_path, dpi=300 ,poppler_path=poppler_path)
        for page in pages:
            image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            if enhance:
                image = apply_clahe_preprocessing(image)
            images.append(image)
    else:
        image = cv2.imread(new_form_path)
        if image is None:
            raise FileNotFoundError("New form image not found.")
        if enhance:
            image = apply_clahe_preprocessing(image)
        images.append(image)
    return images

def register_image(template_img, new_form_img, num_matches=50, ransac_thresh=5.0):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(new_form_img, None)
    
    # Use KNN matching with Lowe's ratio test for robustness
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) < num_matches:
        logging.warning("Found only %d good matches.", len(good_matches))
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:num_matches]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
    if M is None:
        raise Exception("Homography could not be computed!")
    return M

def warp_image(new_form_img, M):
    h_new, w_new = new_form_img.shape[:2]
    corners = np.float32([[0, 0], [w_new, 0], [w_new, h_new], [0, h_new]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)
    [min_x, min_y] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
    [max_x, max_y] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
    new_width = max_x - min_x
    new_height = max_y - min_y
    translation_matrix = np.array([[1, 0, -min_x],
                                   [0, 1, -min_y],
                                   [0, 0, 1]], dtype=np.float32)
    M_adjusted = translation_matrix.dot(M)
    aligned_img = cv2.warpPerspective(new_form_img, M_adjusted, (new_width, new_height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
    return aligned_img, M_adjusted

def overlay_and_crop_annotations(aligned_img, via_data, scale_factor, cropped_folder,
                                 vertical_margin_ratio=0.2, offset_x=0, offset_y=0):
    """
    Overlays annotation boxes on aligned_img and crops them from a clean copy.
    offset_x, offset_y: Additional pixel offsets to shift bounding boxes after scaling.
    """
    aligned_img_for_crop = aligned_img.copy()
    region_count = 0
    for key, metadata in via_data["_via_img_metadata"].items():
        regions = metadata.get("regions", [])
        for region in regions:
            shape_attr = region.get("shape_attributes", {})
            x = shape_attr.get("x", 0)
            y = shape_attr.get("y", 0)
            w = shape_attr.get("width", 0)
            h = shape_attr.get("height", 0)

            # Scale
            x_norm = int(x * scale_factor)
            y_norm = int(y * scale_factor)
            width_norm = int(w * scale_factor)
            height_norm = int(h * scale_factor)

            # Apply offset
            x_norm += offset_x
            y_norm += offset_y

            # Add vertical margin only
            margin_y = max(20, int(height_norm * vertical_margin_ratio))
            new_y_coord = max(0, y_norm - margin_y)
            new_y2_coord = y_norm + height_norm + margin_y
            new_x_coord = x_norm
            new_x2_coord = x_norm + width_norm
            
            # Clamp to image bounds
            h_aligned, w_aligned = aligned_img.shape[:2]
            new_x_coord = max(0, new_x_coord)
            new_y_coord = max(0, new_y_coord)
            new_x2_coord = min(w_aligned, new_x2_coord)
            new_y2_coord = min(h_aligned, new_y2_coord)
            
            region_attr_dict = region.get("region_attributes", {})
            label = list(region_attr_dict.values())[0] if region_attr_dict else "N/A"
            
            # Draw bounding box on aligned image
            cv2.rectangle(aligned_img, (new_x_coord, new_y_coord), (new_x2_coord, new_y2_coord), (0, 255, 0), 2)
            cv2.putText(aligned_img, label, (new_x_coord, new_y_coord - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)
            
            # Crop region if valid
            if new_x_coord < new_x2_coord and new_y_coord < new_y2_coord:
                roi = aligned_img_for_crop[new_y_coord:new_y2_coord, new_x_coord:new_x2_coord]
                if roi.size != 0:
                    cropped_filename = os.path.join(cropped_folder, f"{label}_{region_count}.png")
                    cv2.imwrite(cropped_filename, roi)
                    logging.info("Saved cropped region: %s", cropped_filename)
                else:
                    logging.warning("Skipped empty ROI for %s at region %d", label, region_count)
            else:
                logging.warning("Invalid ROI dimensions for %s at region %d", label, region_count)
            region_count += 1
    return aligned_img

def main(args):
    # Load template
    orig_template_img = cv2.imread(args.template)
    if orig_template_img is None:
        raise FileNotFoundError("Template image not found.")
    template_img = orig_template_img.copy()

    # Load new form images
    new_form_images = load_new_form_images(args.new_form, poppler_path=args.poppler, enhance=args.enhance)

    # Create output folders
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.cropped, exist_ok=True)

    # Compute scale factor from the template's original width
    orig_h, orig_w = template_img.shape[:2]
    scale_factor = args.target_width / float(orig_w)

    # Normalize template
    if args.enhance:
        # Optionally enhance the template too
        template_img = apply_clahe_preprocessing(template_img)
    template_img = normalize_image(template_img, target_width=args.target_width)

    page_num = 1
    for idx, new_form_img in enumerate(new_form_images):
        # Normalize new form
        new_form_img = normalize_image(new_form_img, target_width=args.target_width)
        
        # Register
        M = register_image(template_img, new_form_img, num_matches=args.num_matches, ransac_thresh=args.ransac)
        aligned_img, _ = warp_image(new_form_img, M)
        
        # Optionally save intermediate result
        if args.save_intermediate:
            intermediate_path = os.path.join(args.results, f"aligned_page_{idx+1}.png")
            cv2.imwrite(intermediate_path, aligned_img)
            logging.info("Saved intermediate aligned image: %s", intermediate_path)

        # Load VIA annotations
        with open(args.json, "r") as f:
            via_data = json.load(f)

        # Overlay and crop
        annotated_img = overlay_and_crop_annotations(
            aligned_img, via_data, scale_factor, args.cropped,
            vertical_margin_ratio=args.margin,
            offset_x=args.offset_x,
            offset_y=args.offset_y
        )

        cv2.imshow(f"Aligned Form with Labels (Page {idx+1})", annotated_img)
        cv2.waitKey(0)
        page_num += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document registration and segmentation using VIA annotations.")
    parser.add_argument("--template", type=str, default="data/inference_images/page_1.jpg", help="Path to the template image.")
    parser.add_argument("--new_form", type=str, default="data/inference_images/form_page_1.png", help="Path to the new form (PDF or image).")
    parser.add_argument("--json", type=str, default="data/bulletin_de_soin_labels.json", help="Path to the VIA JSON annotations file.")
    parser.add_argument("--poppler", type=str, default=r"C:\poppler-24.08.0\Library\bin", help="Path to Poppler's bin folder (for PDF conversion).")
    parser.add_argument("--target_width", type=int, default=1000, help="Target width for normalization.")
    parser.add_argument("--num_matches", type=int, default=50, help="Number of good matches to use for registration.")
    parser.add_argument("--ransac", type=float, default=5.0, help="RANSAC reprojection threshold for homography.")
    parser.add_argument("--margin", type=float, default=0.2, help="Vertical margin ratio to add to annotation boxes.")
    parser.add_argument("--results", type=str, default="results", help="Folder to store result images.")
    parser.add_argument("--cropped", type=str, default="cropped_parts", help="Folder to store cropped regions.")
    parser.add_argument("--enhance", action="store_true", help="Apply CLAHE and grayscale preprocessing to images.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate aligned images.")
    parser.add_argument("--offset_x", type=int, default=0, help="Horizontal offset to apply to bounding boxes.")
    parser.add_argument("--offset_y", type=int, default=0, help="Vertical offset to apply to bounding boxes.")
    args = parser.parse_args()
    main(args)
