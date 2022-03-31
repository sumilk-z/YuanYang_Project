from PIL import Image


def cut_pic():
    files = ['bird_female.png', 'bird_male.png', 'obstacle.png']
    for file in files:
        image = Image.open(file)
        new_image = image.resize((120, 80), Image.ANTIALIAS)
        new_image.save("new_" + file)

    image = Image.open("background.jpg")
    new_image = image.resize((1200, 800), Image.ANTIALIAS)
    new_image.save("new_background.jpg")


if __name__ == "__main__":
    cut_pic()
