import csv
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import datetime
import time


# take Image of user
def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    if (l1 == "") and (l2 == ""):
        t = 'Please Enter the your Enrollment Number and Name.'
        try:
            text_to_speech(t)
        except Exception:
            pass
        print(t)
        return False
    elif l1 == '':
        t = 'Please Enter the your Enrollment Number.'
        try:
            text_to_speech(t)
        except Exception:
            pass
        print(t)
        return False
    elif l2 == "":
        t = 'Please Enter the your Name.'
        try:
            text_to_speech(t)
        except Exception:
            pass
        print(t)
        return False
    else:
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                msg = "Cannot open camera. Make sure a webcam is connected and not used by another app."
                try:
                    text_to_speech(msg)
                except Exception:
                    pass
                print(msg)
                return False

            detector = cv2.CascadeClassifier(haarcasecade_path)
            Enrollment = l1
            Name = l2
            sampleNum = 0
            directory = Enrollment + "_" + Name
            path = os.path.join(trainimage_path, directory)
            # keep original behavior: if directory exists, signal FileExistsError
            if os.path.exists(path):
                raise FileExistsError(path)
            os.makedirs(path, exist_ok=False)

            print(f"Starting capture for {Name} (Enrollment: {Enrollment}). Saving to: {path}")
            while True:
                ret, img = cam.read()
                if not ret or img is None:
                    print("Failed to read frame from camera. Stopping.")
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    save_path = os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg")
                    cv2.imwrite(save_path, gray[y: y + h, x: x + w])
                    print(f"Saved image: {save_path}")
                    cv2.imshow("Frame", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Capture stopped by user (q pressed).")
                    break
                elif sampleNum >= 50:
                    print("Reached 50 samples. Stopping capture.")
                    break

            cam.release()
            cv2.destroyAllWindows()

            # ensure StudentDetails dir exists
            sd_dir = os.path.join("StudentDetails")
            os.makedirs(sd_dir, exist_ok=True)
            row = [Enrollment, Name]
            csv_path = os.path.join(sd_dir, "studentdetails.csv")
            with open(csv_path, "a+", newline="") as csvFile:
                writer = csv.writer(csvFile, delimiter=",")
                writer.writerow(row)

            res = "Images Saved for ER No:" + Enrollment + " Name:" + Name
            try:
                message.configure(text=res)
            except Exception:
                print(res)
            try:
                text_to_speech(res)
            except Exception:
                pass
            return True
        except FileExistsError:
            F = "Student Data already exists"
            try:
                text_to_speech(F)
            except Exception:
                pass
            print(F)
            return False


def _console_message():
    class M:
        def configure(self, text=None):
            print(text)

    return M()


def _console_tts(text):
    print("TTS:", text)


def _main():
    parser = argparse.ArgumentParser(description="Capture images for a student (webcam).")
    parser.add_argument("enrollment", help="Enrollment number (id)")
    parser.add_argument("name", help="Student name")
    parser.add_argument("--haar", default="haarcascade_frontalface_default.xml", help="Path to haarcascade xml")
    parser.add_argument("--out", default="TrainingImageLabel", help="Output training images folder")
    args = parser.parse_args()

    TakeImage(args.enrollment, args.name, args.haar, args.out, _console_message(), None, _console_tts)


if __name__ == "__main__":
    _main()
