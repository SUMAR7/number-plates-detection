import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tkinter import PhotoImage
import numpy as np
import cv2
import pytesseract as tess
import pandas as pd


def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if num_contours:
        contour_area = [cv2.contourArea(c) for c in num_contours]
        max_cntr_index = np.argmax(contour_area)

        max_cnt = num_contours[max_cntr_index]
        max_cnt_area = contour_area[max_cntr_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratio_check(max_cnt_area, w, h):
            return plate, None

        final_img = thresh[y:y + h, x:x + w]
        return final_img, [x, y, w, h]

    else:
        return plate, None


def ratio_check(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
        return False
    return True


def is_max_white(plate):
    avg = np.mean(plate)
    if avg >= 115:
        return True
    else:
        return False


def ratio_and_rotation(rect):
    (x, y), (width, height), rect_angle = rect

    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False

    if height == 0 or width == 0:
        return False

    area = height * width
    if not ratio_check(area, width, height):
        return False
    else:
        return True


top = tk.Tk()
top.attributes('-zoomed', True)
top.geometry('900x900')
top.title('Number Plate Recognition')

title = Label(top, text='NIAIS', font=('Arial Bold', 15), background='#E2E2E2')
title.place(x=10, y=10)

name = Label(top, text='Sajjad Umar', font=('Arial Bold', 15), background='#E2E2E2')
name.place(x=10, y=40)

roll = Label(top, text='NIAIS-0983', font=('Arial Bold', 15), background='#E2E2E2')
roll.place(x=10, y=70)

top.iconphoto(True, PhotoImage(file="/home/sajjad/PycharmProjects/number-plates-detection/assets/logo.png"))
img = ImageTk.PhotoImage(Image.open("/home/sajjad/PycharmProjects/number-plates-detection/assets/logo.png"))
top.configure(background='#E2E2E2')
label = Label(top, background='#E2E2E2', font=('arial', 20, 'bold'))
owner_name = Label(top, font=('arial', 20, 'bold'))
model = Label(top, font=('arial', 20, 'bold'))
type = Label(top, font=('arial', 20, 'bold'))
registration = Label(top, font=('arial', 20, 'bold'))

owner_name.pack(side=tk.BOTTOM,pady=15)
model.pack(side=tk.BOTTOM,pady=15)
registration.pack(side=tk.BOTTOM,pady=15)
type.pack(side=tk.BOTTOM,pady=15)


sign_image = Label(top, bd=10)
plate_image = Label(top, bd=10)
owners = pd.read_csv('/home/sajjad/PycharmProjects/number-plates-detection/assets/datasets/owners.csv', sep=",")


def classify(file_path, button):
    res_text = [0]
    res_img = [0]
    img = cv2.imread(file_path)

    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = img2.copy()
    cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    num_contours, hierarchy = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2, num_contours, -1, (0, 255, 0), 1)

    for i, cnt in enumerate(num_contours):

        min_rect = cv2.minAreaRect(cnt)

        if ratio_and_rotation(min_rect):

            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]
            print("Number identified number plate...")

            res_img[0] = plate_img
            cv2.imwrite("/home/sajjad/PycharmProjects/number-plates-detection/assets/result.png", plate_img)

            if is_max_white(plate_img):
                clean_plate, rect = clean2_plate(plate_img)

                if rect:
                    fg = 0
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    res_text[0] = text

                    if text:
                        details = owners[owners["RegistrationNo"] == text.strip()]
                        owner_name.configure(foreground='#011638', text=f"Owner Name: {details['Name'].values[0]}")
                        registration.configure(foreground='#011638',
                                               text=f"Registration No: {details['RegistrationNo'].values[0]}")
                        model.configure(foreground='#011638', text=f"Model: {details['Make'].values[0]}")
                        type.configure(foreground='#011638', text=f"Type: {details['Type'].values[0]}")
                        print(f"details: {details}")
                        print("Number Detected Plate Text : ", text.strip())
                        break

    label.configure(foreground='#011638', text=res_text[0].strip())
    label.place(x=1300, y=450)
    uploaded = Image.open("/home/sajjad/PycharmProjects/number-plates-detection/assets/result.png")
    im = ImageTk.PhotoImage(uploaded)
    plate_image.configure(image=im)
    plate_image.image = im
    plate_image.pack(side=tk.RIGHT, padx=350)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path, classify_b), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
    classify_b.place(x=250, y=700)


def upload_image():
    try:
        file_path = filedialog.askopenfilename(initialdir='/home/sajjad/PycharmProjects/number-plates-detection/assets/test_samples')
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
upload.pack(side=tk.BOTTOM, pady=10)
sign_image.pack()
sign_image.place(x=70, y=200)

label.pack()
label.place(x=500, y=220)
heading = Label(top, image=img)
heading.configure(background='#E2E2E2', foreground='#364156')
heading.pack(side=TOP, pady=10)

starting_image = Image.open("/home/sajjad/PycharmProjects/number-plates-detection/assets/niais.png")
im = ImageTk.PhotoImage(starting_image)
sign_image.configure(image=im)
sign_image.image = im
label.configure(text='')

top.mainloop()
