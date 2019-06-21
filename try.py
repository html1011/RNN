import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
import matplotlib.pyplot as plt
from tkinter import *
from skimage.transform import rescale, resize, downscale_local_mean
# from math import sin
import numpy as np
from PIL import Image
import os
mnist = tf.keras.datasets.mnist
pathName = "drawing.h5"
def matmul_on_gpu(n):
  if n.type == "MatMul":
    print("Using GPU")
    return "/device:GPU:0"
  else:
    return "/cpu:0"
# pathName = os.path.dirname(pathName)
with tf.device("/device:GPU:0"):
    print(pathName)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(y_train[0])
    print(x_train.shape)
    x_train = x_train/255
    x_test = x_test/255
    try:
        # Loading model
        print("Loaded model")
        model = load_model(pathName)
    except:
        # plt.imshow(x_train[1])
        # plt.show()
        model = Sequential()
        model.add(CuDNNLSTM(28, input_shape=(x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation="softmax"))
        optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        model.compile(loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"])
        model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
        model.save(pathName)

    master = Tk()
    WIDTH, HEIGHT = x_train[0].shape
    SCALE = 3
    w = Canvas(master, width=WIDTH * SCALE, height=HEIGHT * SCALE)
    img = PhotoImage(width=WIDTH * SCALE, height=HEIGHT * SCALE)
    # Draw same image in parallel using PIL

    # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
    img1 = Image.new( 'RGB', (WIDTH * SCALE, HEIGHT * SCALE), (255, 255, 255)) # create a new black image
    # pixels = img.load() # create the pixel map

    print(WIDTH, HEIGHT)
    def callback1(event):
        w.focus_set()
        img.put("#000000", (event.x, event.y))
        img1.putpixel((event.x, event.y), (0, 0, 0))
        # print(img.get(event.x, event.y))
        # print("List: ", img.get(event.x + 10, event.y))
        # print(list(img.get_data(event.x, event.y)))
    def callback2(event):
        newInput = []
        loadIt = img1.load()
        for i in range(HEIGHT * SCALE):
            newInput1 = []
            for ii in range(WIDTH * SCALE):
                # Calculate an average of the amounts.
                avg = loadIt[event.x, event.y][0]
                if ii % 10 == 0:
                    print(avg)
                newInput1.append(avg)
                # While we do this, we delete the current image.
                # img.put("#FFFFFF", (i, ii))
            newInput.append(newInput1)
            newInput1 = []
        newInput = np.array(newInput)
        # newInput = newInput / 255
        plt.imshow(newInput)
        plt.show()
        newConcate = []
        # for i in range(0, newInput.shape[0], SCALE):
        #     # Looking at the height; we create a mini kernel and move around.
        #     newConcate1 = []
        #     for ii in range(0, newInput.shape[1], SCALE):
        #         # Looking at the width
        #         newConcate1.append(np.max(newInput[ii:ii+SCALE, i:i+SCALE]))
        #     newConcate.append(newConcate1)
        # print(newConcate[5])
        newConcate = np.array(newConcate)
        plt.imshow(newConcate)
        plt.show()
        print(newConcate.shape)
        # print(model.get_weights())
        # guess = model.predict(np.array([newConcate]))
        # print(guess)
        print(newConcate.shape, x_train.shape)
        w.focus_set()
        # w.delete("all")
        # w.create_rectangle(0, 0, WIDTH * SCALE, HEIGHT * SCALE, color="white")
        # img = PhotoImage(width=WIDTH * SCALE, height=HEIGHT * SCALE)
        print(np.argmax(guess[0]) + 1)
        w.create_text(10, 10, fill="black", font="Monospace 8", text=str(np.argmax(guess[0]) + 1))
        w.create_image(((WIDTH * SCALE)/2, (HEIGHT * SCALE)/2), image=img, state="normal")

    w.create_image(((WIDTH * SCALE)/2, (HEIGHT * SCALE)/2), image=img, state="normal")

    w.bind("<B1-Motion>", callback1)
    w.bind("<ButtonRelease-1>", callback2)
    w.pack()
    # w.create_line(0, 0, 200, 100)
    # w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

    # w.create_rectangle(50, 25, 150, 75, fill="blue")

    mainloop()