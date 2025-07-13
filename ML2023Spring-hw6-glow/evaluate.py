from ultralytics import YOLO
from pytorch_fid import fid_score
import os

def calculate_fid(real_images_path, generated_images_path):
    fid = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], batch_size=50, device='cuda', dims=2048)
    return fid

def calculate_afd(generated_images_path, save=True):
    yolov8_animeface = YOLO('./cpt/yolov8x6_animeface.pt')
    results = yolov8_animeface.predict(generated_images_path, save=save, conf=0.8, iou=0.8, imgsz=64)

    anime_faces_detected = 0
    total_images = len(results)

    for result in results:
        if len(result.boxes) > 0:
            anime_faces_detected += 1

    afd_score = anime_faces_detected / total_images
    return afd_score

if __name__ == '__main__':
    real_images_path = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw6/faces/faces'
    generated_images_path = './output'
    afd = calculate_afd(generated_images_path)
    fid = calculate_fid(real_images_path, generated_images_path)
    print(f'FID:{fid:.3f}')
    print(f'AFD:{afd:.3f}')