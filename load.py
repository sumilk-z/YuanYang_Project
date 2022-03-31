#!/usr/bin/python3
# coding = UTF-8

import pygame
import os.path as osp

path_file = osp.abspath(__file__)
path_images = osp.join(path_file, '..', 'images')


def load_bird_male():
    obj = 'new_bird_male.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)


def load_bird_female():
    obj = 'new_bird_female.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)


def load_background():
    obj = 'new_background.jpg'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)


def load_obstacle():
    obj = 'new_obstacle.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)