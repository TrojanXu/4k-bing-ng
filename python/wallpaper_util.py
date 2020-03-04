import ctypes
import os
import sys

def set_windows_wallpaper(pic_path):
    ctypes.windll.user32.SystemParametersInfoW(20, 0, pic_path , 0)
    print('Wallpaper is set.')


def set_linux_wallpaper(pic_path):
    os.system(''.join(['gsettings set org.gnome.desktop.background picture-uri file://', pic_path]))
    print('Wallpaper is set.')


def set_wallpaper(pic_path):
    if sys.platform.startswith('win32'):
        set_windows_wallpaper(pic_path)
    elif sys.platform.startswith('linux'):
        set_linux_wallpaper(pic_path)
    else:
        print('OS not supported.')