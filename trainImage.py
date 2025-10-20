import csv
import os
import re
import argparse
import cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image


# Train Image
def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message,text_to_speech):
    # Create LBPH recognizer with a few fallbacks depending on cv2 build
    recognizer = None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        # Some cv2 builds put the contrib modules under cv2.cv2
        try:
            import cv2 as _cv

            if hasattr(_cv, 'cv2') and hasattr(_cv.cv2, 'face'):
                recognizer = _cv.cv2.face.LBPHFaceRecognizer_create()
        except Exception:
            pass

    if recognizer is None:
        msg = (
            'cv2.face.LBPHFaceRecognizer_create() is not available in your OpenCV build. '\
            'Install a contrib build: pip install opencv-contrib-python'
        )
        try:
            message.configure(text=msg)
        except Exception:
            print(msg)
        try:
            text_to_speech(msg)
        except Exception:
            pass
        return False
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, Id = getImagesAndLables(trainimage_path)
    if not faces or not Id:
        msg = f"No training images found in '{trainimage_path}'. Aborting training."
        try:
            message.configure(text=msg)
        except Exception:
            print(msg)
        try:
            text_to_speech(msg)
        except Exception:
            pass
        return False

    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    res = "Image Trained successfully"  # +",".join(str(f) for f in Id)
    try:
        message.configure(text=res)
    except Exception:
        print(res)
    try:
        text_to_speech(res)
    except Exception:
        pass
    return True


def getImagesAndLables(path):
    """Discover image files under `path` and return (faces, Ids).

    Behavior:
    - Accept images directly inside `path` or inside immediate subfolders.
    - Accept common image extensions.
    - Extract an integer ID from the filename using the pattern *_<id>_*.ext or the first integer found.
    - Skip files that cannot be parsed or opened.
    """
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return [], []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pgm"}

    # Collect candidate image file paths
    candidates = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in exts:
            candidates.append(full)
        elif os.path.isdir(full):
            # gather files inside subfolder
            for fn in sorted(os.listdir(full)):
                ffull = os.path.join(full, fn)
                if os.path.isfile(ffull) and os.path.splitext(fn)[1].lower() in exts:
                    candidates.append(ffull)

    faces = []
    Ids = []
    id_re = re.compile(r"_(\d+)_")
    int_re = re.compile(r"(\d+)")

    for img_path in candidates:
        try:
            pilImage = Image.open(img_path).convert("L")
            imageNp = np.array(pilImage, "uint8")
        except Exception as e:
            print(f"Skipping unreadable image {img_path}: {e}")
            continue

        # Try to extract ID from filename using common patterns
        fname = os.path.split(img_path)[-1]
        m = id_re.search(fname)
        Id = None
        if m:
            Id = int(m.group(1))
        else:
            m2 = int_re.search(fname)
            if m2:
                Id = int(m2.group(1))

        if Id is None:
            print(f"Could not parse ID from filename, skipping: {fname}")
            continue

        faces.append(imageNp)
        Ids.append(Id)

    return faces, Ids


def _console_message():
    class M:
        def configure(self, text=None):
            print(text)

    return M()


def _console_tts(text):
    print("TTS:", text)


def _main():
    parser = argparse.ArgumentParser(description="Train LBPH face recognizer from dataset.")
    parser.add_argument("--haar", default="haarcascade_frontalface_default.xml", help="Path to haarcascade xml")
    parser.add_argument("--images", default="TrainingImageLabel", help="Path to training images folder")
    parser.add_argument("--out", default="TrainingImageLabel/Trainner.yml", help="Output trained model path")
    args = parser.parse_args()

    print("haar:", args.haar)
    print("images:", args.images)
    print("out:", args.out)

    ok = TrainImage(args.haar, args.images, args.out, _console_message(), _console_tts)
    if ok:
        print("Training completed and model saved to:", args.out)
    else:
        print("Training did not run. See messages above.")


if __name__ == "__main__":
    _main()
